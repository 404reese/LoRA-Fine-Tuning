# LoRA Fine-Tuning

Experiments in fine-tuning TinyLlama with LoRA (Low-Rank Adaptation) —
freezing the base model and training small adapter matrices on top of it
for specific tasks.

## Concepts

- [How LoRA actually works](./how-lora-works.md) — where LoRA attaches
  (attention projections, not embeddings), the math behind `W' = W + BA`,
  why it's called "low-rank," and how much cheaper it is than full
  fine-tuning.

## Experiments

- [Math fine-tuning (GSM8K)](./math-finetuning.md) — teaching TinyLlama to
  answer grade-school math word problems by attaching LoRA adapters to its
  Query/Value attention projections and training on 200 GSM8K examples.
  Notebook: [`Fine-Tuning_TinyLlama_Math.ipynb`](./Fine-Tuning_TinyLlama_Math.ipynb)
