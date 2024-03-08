from typing import List, Tuple
import json


class MCQ:
    def __init__(self, question: str, choices: List[str], correct: str):
        self.question = question
        self.choices = choices
        self.correct = correct

    def __str__(self):
        q = self.question

        for idx, choice in enumerate(self.choices):
            letter = chr(97 + idx)
            q += f"\n{letter}) {choice}"

        q += "\nAnswer: "

        return q

    @staticmethod
    def from_json(json_input: str) -> "MCQ":
        json_dict = json.loads(json_input)
        question = json_dict["question"]["stem"]
        choices = [choice["text"] for choice in json_dict["question"]["choices"]]
        label = json_dict["answerKey"]

        return MCQ(question, choices, label)


class MCQDataset:
    def __init__(self, path: str, few_shot: bool = False):
        self.mcqs = []

        with open(path, "r") as f:
            for line in f:
                mcq = MCQ.from_json(line)
                self.mcqs.append(mcq)

        if few_shot:
            self.few_shot = self.mcqs[:3]
            self.mcqs = self.mcqs[3:]

    def few_shot_prompt(self) -> str:
        prompt = ""
        for mcq in self.few_shot:
            prompt += mcq.question + "\n"
            for idx, choice in enumerate(mcq.choices):
                letter = chr(97 + idx)
                prompt += f"{letter}) {choice}\n"
            prompt += "Answer: " + mcq.correct + "\n\n"
        return prompt

    def __str__(self):
        return f"MCQs: {self.mcqs}"

    def __len__(self):
        return len(self.mcqs)

    def __getitem__(self, idx: int) -> MCQ:
        return self.mcqs[idx]

    def __iter__(self):
        return iter(self.mcqs)


if __name__ == "__main__":
    mcq_dataset = MCQDataset("./data/train.jsonl", few_shot=True)

    print(mcq_dataset.few_shot_prompt())
