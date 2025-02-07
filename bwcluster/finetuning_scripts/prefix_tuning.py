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
# This script directly works with past_key_values

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

