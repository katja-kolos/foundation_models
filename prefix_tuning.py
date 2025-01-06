import torch

from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DynamicCache

# Prefix Tuning Components are defined here
# Inspiration: https://arxiv.org/pdf/2101.00190
# Idea: avoid fine-tuning whole (generative) model; instead, learn a prefix for the specific task.
# Prefix is of fixed length, e.g. 10 tokens, that are learnt from train dataset.
# Current design decisions: 
# 1) these tokens are prepended to normal input with the hope of better explaining the taski.
# 2) the tokens do not directly map to some existing vocabulary items: the additional MLP layer generates embeddings and not input_ids, those embeddings might not correspond to anything meaningful
# we might further study if initializing from existing words leads to better results
# Further work: directly work with past_key_values instead of explicit prepending
"""
class PrefixTuning(nn.Module):
    def __init__(self, config, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = config.hidden_size
        #P'_theta
        self.prefix_embedding = nn.Embedding(prefix_length, config.hidden_size // 2)
        #MLP_theta
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size, dtype=torch.bfloat16),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size, dtype=torch.bfloat16)
        )

    def forward(self, inputs_embeds):
        batch_size = inputs_embeds.size(0)
        prefix = self.prefix_embedding(inputs_embeds)
        prefix = self.mlp(prefix)
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
        # Note: Embeddings can be made up by the MLP + paper uses them as past_key_values.
        return torch.cat([prefix, inputs_embeds], dim=1)


class PrefixTuningModel(nn.Module):
    def __init__(self, model, tokenizer, prefix_length=10):
        super().__init__()
        self.model = model
        self.freeze_main_model()
        self.tokenizer = tokenizer
        self.prefix_tuning = PrefixTuning(self.model.config, prefix_length)

    def freeze_main_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, inputs, labels):
        inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        # Add Prefix
        inputs_embeds = self.prefix_tuning(inputs_embeds)

        # Modify attention mask for prefix
        prefix_mask = torch.ones((inputs["input_ids"].size(0), self.prefix_tuning.prefix_length), device=inputs["input_ids"].device)
        attention_mask = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)

        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, pixel_values=inputs["pixel_values"], labels=labels)

    def generate(self, inputs, max_new_tokens):
        inputs_embeds = self.model.get_input_embeddings()(inputs["input_ids"])
        inputs_embeds = self.prefix_tuning(inputs_embeds)
        prefix_mask = torch.ones((inputs["input_ids"].size(0), self.prefix_tuning.prefix_length),
                                 device=inputs["input_ids"].device)
        attention_mask = torch.cat([prefix_mask, inputs["attention_mask"]], dim=1)
        return self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, pixel_values=inputs["pixel_values"], max_new_tokens=max_new_tokens)"""


class PrefixTuningPastKeyValues(nn.Module):
    def __init__(self, match_n_layer, match_n_head, n_embd, device, prefix_length=30):
        super().__init__()
        self.device = device
        self.prefix_length = prefix_length
        self.input_tokens = torch.arange(prefix_length).long()
        self.match_n_layer = match_n_layer
        self.match_n_head = match_n_head
        self.match_n_embd = n_embd // match_n_head
        self.n_embd = n_embd
        #P'_theta
        self.prefix_embedding = nn.Embedding(self.prefix_length, self.n_embd, dtype=torch.bfloat16)
        #MLP_theta
        self.mid_dim = self.n_embd
        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim, dtype=torch.bfloat16),
            nn.Tanh(),
            nn.Linear(self.mid_dim, (self.match_n_layer * 2 * self.n_embd), dtype=torch.bfloat16)
        )

    def forward(self, bsz):
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1).to(self.device)
        prefix = self.prefix_embedding(input_tokens)
        past_key_values = self.mlp(prefix)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, 2*self.match_n_layer, self.match_n_head, self.match_n_embd)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        for (key, val) in past_key_values:
            print(f"Layer: {key.shape}, {val.shape}")
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        return past_key_values


class PrefixTuningModelPastKeyValues(nn.Module):
    def __init__(self, model, match_n_layer, match_n_head, n_embd, device, prefix_length=30):
        super().__init__()
        self.model = model
        self.freeze_main_model()
        self.prefix_tuning = PrefixTuningPastKeyValues(match_n_layer, match_n_head, n_embd, device, prefix_length)

    def freeze_main_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, inputs, labels):
        bsz = inputs["input_ids"].size(0)
        tgt_attn = torch.ones((bsz, self.prefix_tuning.prefix_length), device=inputs["input_ids"].device)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], tgt_attn], dim=1)
        past_key_values = self.prefix_tuning(bsz)
        return self.model(**inputs, labels=labels, past_key_values=past_key_values)


class PrefixDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return row['input'], row['message'], row['image'], row['solution']

def prefix_collate(batch):
    input, message, image, y = zip(*batch)
    return input, message, image, y
