# Mistral-7B Medical Instruct — QLoRA Fine-tuning Pipeline

Fine-tuning pipeline for Mistral-7B on medical Q&A using QLoRA on Google Colab.

## 🤗 Published Model
👉 [AbhijithSK/mistral-7b-medical-instruct](https://huggingface.co/AbhijithSK/mistral-7b-medical-instruct)

## Training Results
| Metric | Value |
|--------|-------|
| Final Loss | 0.022 |
| Token Accuracy | 98.9% |
| Epochs | 3 |
| Training Time | ~2h 50min on T4 GPU |
| Trainable Params | 13.6M / 7.25B (0.19%) |

## Method
- Base model: mistralai/Mistral-7B-Instruct-v0.2
- Dataset: medalpaca/medical_meadow_pubmed_causal
- Quantization: 4-bit NF4 (QLoRA)
- LoRA rank: 16, alpha: 32

## Quick Start
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = PeftModel.from_pretrained(model, "AbhijithSK/mistral-7b-medical-instruct")
tokenizer = AutoTokenizer.from_pretrained("AbhijithSK/mistral-7b-medical-instruct")
```