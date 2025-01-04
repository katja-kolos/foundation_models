import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class PrefixTuning(nn.Module):
    def __init__(self, config, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        self.hidden_size = config.hidden_size
        #P'_theta
        self.prefix_param = nn.Parameter(torch.randn(prefix_length, config.hidden_size // 2))
        #MLP_theta
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size // 2, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, inputs_embeds):
        batch_size = inputs_embeds.size(0)
        prefix = self.prefix_embeddings
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
        return self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, pixel_values=inputs["pixel_values"], max_new_tokens=max_new_tokens)