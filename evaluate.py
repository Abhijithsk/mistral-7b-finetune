import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import evaluate
from data.prepare_dataset import load_config

def merge_and_evaluate():
    config = load_config()
    print("Loading base model for merging...")
    base = AutoModelForCausalLM.from_pretrained(
        config["model_id"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, config["output_dir"])
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])

    model.save_pretrained("./mistral-7b-merged")
    tokenizer.save_pretrained("./mistral-7b-merged")
    print("Merged model saved.")

    rouge = evaluate.load("rouge")
    dataset = load_dataset(config["dataset"]["name"], split="train[5000:5100]")
    predictions, references = [], []

    for example in dataset:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        pred = pred.split("### Response:\n")[-1].strip()
        predictions.append(pred)
        references.append(example["output"])

    scores = rouge.compute(predictions=predictions, references=references)
    print("\n── Evaluation Results ──")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    merge_and_evaluate()