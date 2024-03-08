import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import save_file, load_file

from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM


class TrainConfig:
    lr: float
    num_epochs: int


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
        self.toks = [f"MIDDLE-{i}" for i in range(num_toks)]

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

    def set_tensors(self, embedding: torch.tensor, out: torch.tensor):
        self.embeddings = embedding
        self.out = out

    def __str__(self):
        return f"Name: {self.name}, Prefix: {self.prefix}, Suffix: {self.suffix}, Num Toks: {self.num_toks}"


class LearnedPromptModel:
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None

    def __init__(
        self,
        model: str,
        tokenizer: str,
        prompts: List[LearnedPrompt] = [],
    ):
        super(LearnedPromptModel, self).__init__(model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.prompts: List[LearnedPrompt] = prompts

        model.to(self.device)

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
        pass

    def nearest_tokens(self, prompt: LearnedPrompt) -> List[str]:
        nearest_tokens = []
        for i in range(len(prompt.embeddings)):
            embedding = prompt.embeddings[i]
            max_distance = 0
            nearest_token = ""
            for j in range(len(self.tokenizer)):
                token = self.tokenizer.decode(j)
                distance = torch.norm(
                    embedding - self.model.model.embed_tokens.weight.data[j]
                )
                if distance > max_distance:
                    max_distance = distance
                    nearest_token = token
            nearest_tokens.append(nearest_token)

        return nearest_tokens

    def train_prompt(
        self,
        prompt: LearnedPrompt,
        config: TrainConfig,
    ):
        self.model.train()

        num_addition_embeddings = len(prompt.toks)

        new_embeddings = nn.Embedding(
            self.vocab_size + num_addition_embeddings, self.d_model
        )
        nn.init.xavier_uniform_(new_embeddings.weight.data)
        new_embeddings.weight.data[: self.vocab_size] = (
            self.model.model.embed_tokens.weight.data
        )
        if prompt.embeddings is not None:
            new_embeddings.weight.data[self.vocab_size :] = prompt.embeddings
        else:
            prompt.embeddings = new_embeddings.weight.data[self.vocab_size :]

        self.model.model.embed_tokens = new_embeddings

        out_temp = nn.Linear(self.d_model, self.vocab_size + num_addition_embeddings)
        out_temp.weight.data[: self.vocab_size] = self.model.lm_head.weight.data

        if prompt.out is not None:
            out_temp.weight.data[self.vocab_size :] = prompt.out
        else:
            prompt.out = out_temp.weight.data[self.vocab_size :]

        self.model.lm_head = out_temp

        self.prompts.append(prompt)

        self.tokenizer.add_tokens(prompt.toks)

        lr, num_epochs = config.lr, config.num_epochs

        for epoch in range(num_epochs):
            input_str = (
                prompt.prefix + " " + " ".join(prompt.toks) + " " + prompt.suffix
            )
            input = self.tokenizer.encode(input_str, return_tensors="pt")
            input = input.to(self.device)

            logits = self.model.forward(input).logits

            # Adjusting the target and logits for proper loss calculation
            target = input[:, 1:]
            logits = logits[:, :-1, :]
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size + num_addition_embeddings),
                target.view(-1),
            )

            loss.backward()

            self.model.model.embed_tokens.weight.data[self.vocab_size :] -= (
                (self.model.model.embed_tokens.weight.grad.data[self.vocab_size :]) * lr
            )

            self.model.lm_head.weight.data[self.vocab_size :] -= (
                (self.model.lm_head.weight.grad.data[self.vocab_size :]) * lr
            )

            for param in self.model.model.parameters():
                param.grads = None

            prompt.embeddings = self.model.model.embed_tokens.weight.data[
                self.vocab_size :
            ]
            prompt.out = self.model.lm_head.weight.data[self.vocab_size :]

            print("Epoch:", epoch, "Loss:", loss.item())
        return prompt


if __name__ == "__main__":
    model = LearnedPromptModel(
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2",
    )

    prompt = LearnedPrompt(
        "test",
        "The sky is",
        "blue",
        3,
    )

    config = TrainConfig(lr=0.1, num_epochs=10)

    prompt = model.train_prompt(prompt, config)

    print(prompt)
