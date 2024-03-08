import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import save_file, load_file

from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


class LearnedPromptModel:
    model = None
    tokenizer = None

    def __init__(self, model: str, tokenizer: str, num_middle_toks: int):
        super(LearnedPromptModel, self).__init__(model)
        self.model = model
        self.tokenizer = tokenizer
        self.num_middle_toks = num_middle_toks
        self.prompts: List[LearnedPrompt] = []

    def get_prompt(self, name: str) -> LearnedPrompt:
        for prompt in self.prompts:
            if prompt.name == name:
                return prompt
        return None

    def generate(self, name):
        pass


class LearnedPrompt:
    def __init__(
        self,
        name: str,
        prefix: str,
        suffix: str,
        num_toks: int,
        embeddings: torch.tensor = None,
        out: torch.tensor = None,
    ):
        self.name = name
        self.prefix = prefix
        self.suffix = suffix
        self.num_toks = num_toks

        self.embeddings = embeddings
        self.out = out

    def update(self, embeddings: torch.tensor, out: torch.tensor):
        self.embeddings = embeddings
        self.out = out

    def __str__(self):
        return f"Name: {self.name}, Prefix: {self.prefix}, Suffix: {self.suffix}, Num Toks: {self.num_toks}"
