import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from data import MCQDataset

from typing import List, Tuple

from safetensors.torch import save_file, load_file

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
NUM_MIDDLE_TOKS = 5
VOCAB_SIZE = 32000
D_MODEL = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = MCQDataset("./data/train.jsonl", few_shot=True)
testset = MCQDataset("./data/test.jsonl", few_shot=True)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

middle_tokens = [f"[MIDDLE-{i}]" for i in range(NUM_MIDDLE_TOKS)]

# expand the tokenizer to include the middle tokens, and expand the model to include the new tokens
tokenizer.add_tokens(middle_tokens)

new_embeddings = nn.Embedding(VOCAB_SIZE + NUM_MIDDLE_TOKS, D_MODEL)
nn.init.xavier_uniform_(new_embeddings.weight.data)
new_embeddings.weight.data[:VOCAB_SIZE] = model.model.embed_tokens.weight.data

out_temp = nn.Linear(D_MODEL, VOCAB_SIZE + NUM_MIDDLE_TOKS, bias=False)
nn.init.xavier_uniform_(out_temp.weight.data)
out_temp.weight.data[:VOCAB_SIZE] = model.lm_head.weight.data

model.model.embed_tokens = new_embeddings
model.lm_head = out_temp

model.to(device)


def benchmark():
    correct = 0
    for idx in range(len(testset)):
        answer, correct_answer = sample(idx, dataset=testset)
        if answer == correct_answer:
            correct += 1
        print(f"Correct: {correct}/{idx + 1}", end="\r")
    return correct / len(testset)


def sample(idx, dataset=dataset):
    model.eval()
    prompt = (
        " ".join(middle_tokens)
        + " "
        + dataset.few_shot_prompt()
        + " "
        + str(dataset[idx])
    )
    input = tokenizer.encode(prompt, return_tensors="pt")
    input = input.to(device)

    preds = model.generate(
        input,
        do_sample=True,
        max_new_tokens=1,
        eos_token_id=2,
    )

    words = tokenizer.batch_decode(preds)[0]

    answer = words.split()[-1]
    correct_answer = dataset[idx].correct

    return answer, correct_answer


print(benchmark())
exit()

for epoch in range(NUM_EPOCHS):
    for mcq in dataset:
        prompt = dataset.few_shot_prompt() + str(mcq) + " ".join(middle_tokens)
        input = tokenizer.encode(prompt, return_tensors="pt")
        input = input.to(device)

        logits = model.forward(input).logits

        target = input[:, 1:]
        logits = logits[:, :-1, :]
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE + NUM_MIDDLE_TOKS),
            target.view(-1),
        )

        loss.backward()

        model.model.embed_tokens.weight.data[VOCAB_SIZE:] -= (
            (model.model.embed_tokens.weight.grad.data[VOCAB_SIZE:]) * LEARNING_RATE
        )

        model.lm_head.weight.data[VOCAB_SIZE:] -= (
            (model.lm_head.weight.grad.data[VOCAB_SIZE:]) * LEARNING_RATE
        )

        for param in model.model.parameters():
            param.grads = None

        print(f"Epoch {epoch} Loss: {loss.item()}")
