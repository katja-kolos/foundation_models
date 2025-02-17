import torch

from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Prefix Tuning Components are defined here
# Inspiration: https://arxiv.org/pdf/2101.00190
# Idea: avoid fine-tuning whole (generative) model; instead, learn a prefix for the specific task.
# Prefix is of fixed length, e.g. 10 tokens, that are learnt from train dataset.
# Current design decisions: 
# 1) these tokens are prepended to normal input with the hope of better explaining the taski.
# 2) the tokens do not directly map to some existing vocabulary items: the additional MLP layer generates embeddings and not input_ids, those embeddings might not correspond to anything meaningful
# we might further study if initializing from existing words leads to better results
# Further work: directly work with past_key_values instead of explicit prepending

class PrefixTuning(nn.Module):
    def __init__(self, config, prefix_length=10):
        super().__init__()
        self.prefix_length = int(prefix_length)
        self.hidden_size = int(config.hidden_size)
        #P'_theta
        #self.prefix_param = nn.Parameter(torch.randn(prefix_length, config.hidden_size // 2).to(device='cuda', dtype=torch.bfloat16))
        #self.prefix_embedding = nn.Embedding(prefix_length, config.hidden_size // 2).to(device='cuda', dtype=torch.bfloat16)
        self.prefix_dim = int(config.hidden_size // 2) #floor division should return int but just in case because I don't know anymore what is wrong
        #print(f"self.prefix_length: {self.prefix_length}, {type(self.prefix_length)}")
        #print(f"self.prefix_dim: {self.prefix_dim}, {type(self.prefix_dim)}")
        self.prefix_embedding = nn.Parameter(torch.randn(self.prefix_length, self.prefix_dim).to(device='cuda', dtype=torch.bfloat16))
        #print(f"self.prefix_embedding.size()")
        #print(self.prefix_embedding.size())
        #MLP_theta
        self.mlp = nn.Sequential(
            nn.Linear(self.prefix_dim, self.hidden_size, dtype=torch.bfloat16),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size, dtype=torch.bfloat16)
        )

    def forward(self, input_embeds):
        #print("input_embeds")
        #print(input_embeds)
        #print(type(input_embeds))
        batch_size = input_embeds.size(0)
        #print(f"batch_size: {batch_size}")
        prefix = self.prefix_embedding
        prefix = self.mlp(prefix)
        prefix = prefix.unsqueeze(0).expand(batch_size, -1, -1)
        # Note: Embeddings can be made up by the MLP + paper uses them as past_key_values.
        return torch.cat([prefix, input_embeds], dim=1)


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
        return self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, pixel_values=inputs["pixel_values"], max_new_tokens=max_new_tokens)

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
