import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
import PIL


class SoftPrompting:
    @classmethod
    def from_pretrained(
            cls,
            model,
            soft_prompt_path: str = None,
            n_tokens: int = None,
            initialize_from_vocab: bool = True,
            random_range: float = 0.5,
            **kwargs,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model

    def set_soft_prompt_embeds(self, soft_prompt_path: str) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soft_prompt = torch.load(soft_prompt_path, map_location=device)
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
            self, n_tokens: int = 20, initialize_from_vocab: bool = True, random_range: float = 0.5
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.transformer.wte.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(n_tokens, self.config.n_embd).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, self.config.n_embd)
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        learned_embeds = self.soft_prompt.weight.unsqueeze(0).expand(
            inputs_embeds.size(0), -1, -1
        )

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            use_cache=None,
            return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids)

        if labels is not None:
            labels = self._extend_labels(labels)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class MultimodalSoftPrompting(SoftPrompting):
    def forward(
            self,
            input_ids=None,
            images=None,
            attention_mask=None,
            labels=None,
            use_cache=None,
            return_dict=None,
    ):
        # Soft Prompting for texts
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids)

        if images is not None:
            if isinstance(images, PIL.Image.Image):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),  # change for model needs
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                images = transform(images).unsqueeze(0)

            image_features = self.image_encoder(images)

        if labels is not None:
            labels = self._extend_labels(labels)
        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask)

        multimodal_features = torch.cat([inputs_embeds, image_features], dim=1) # change to format for model

        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=multimodal_features,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )