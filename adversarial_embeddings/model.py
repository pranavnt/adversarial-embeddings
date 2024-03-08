import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import save_file, load_file

from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM


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

    def save(self, path: str):
        state_dict = {
            "name": self.name,
            "prefix": self.prefix,
            "suffix": self.suffix,
            "num_toks": self.num_toks,
            "embeddings": self.embeddings,
            "out": self.out,
        }

        save_file(state_dict, path)

    def nearest_token(self, token: str, k: int = 1) -> List[str]:
        pass

    def __str__(self):
        return f"Name: {self.name}, Prefix: {self.prefix}, Suffix: {self.suffix}, Num Toks: {self.num_toks}"


class LearnedPromptModel:
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None

    def __init__(
        self,
        model: str,
        tokenizer: str,
        num_middle_toks: int,
        prompts: List[LearnedPrompt] = [],
    ):
        super(LearnedPromptModel, self).__init__(model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.num_middle_toks = num_middle_toks
        self.middle_toks = [f"[MIDDLE-{i}]" for i in range(num_middle_toks)]
        self.prompts: List[LearnedPrompt] = []

        model.to(self.device), tokenizer.to(self.device)

        self.vocab_size, self.d_model = self.model.lm_head.weight.shape

        assert self.vocab_size == len(tokenizer)

    def get_prompt(self, name: str) -> LearnedPrompt:
        for prompt in self.prompts:
            if prompt.name == name:
                return prompt
        return None

    def generate(self, name, input, max_length=50):
        self.model.eval()
        prompt = self.get_prompt(name)

        if prompt is None:
            self.to_base_model()
        else:
            self.to_prompt_model(prompt)

        input = input + prompt.middle_toks
        input = self.tokenizer.encode(input, return_tensors="pt")
        input = input.to(self.device)

        output = ""

        for _ in range(max_length):
            outputs = self.model(input, return_dict=True)
            logits = outputs.logits
            softmax_logits = F.softmax(logits, dim=-1)
            prediction = torch.argmax(softmax_logits, dim=-1)
            input = torch.cat((input, prediction[:, -1].unsqueeze(-1)), dim=-1)
            output += self.tokenizer.decode(prediction[:, -1].item())

        return output

    def to_base_model(self):
        self.model.model.embed_tokens.weight = self.model.embed_tokens.weight[
            : self.vocab_size
        ]
        self.model.lm_head.weight = self.model.lm_head.weight[self.vocab_size :]

    def to_prompt_model(self, prompt: LearnedPrompt):
        self.model.model.embed_tokens.weight = torch.cat(
            (self.model.model.embed_tokens.weight, prompt.embeddings), dim=0
        )
        self.model.lm_head.weight = torch.cat(
            (self.model.lm_head.weight, prompt.out), dim=0
        )

    def prob_completion(
        self, prompt: LearnedPrompt, completion: str
    ) -> Tuple[float, float]:
        """
        Computes the probability of the completion given the prompt, by multiplying the probabilities of each token in the completion.

        Args:
            prompt (LearnedPrompt): LearnedPrompt object to be used
            completion (str): Completion to be used

        Returns:
            Tuple[float, float]: Probability of the completion given the prompt, and the probability of the completion given the base model
        """
        print("UNIMPLEMENTED")
        pass
