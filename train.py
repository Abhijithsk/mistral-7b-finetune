import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from data.prepare_dataset import get_dataset, load_config

def load_model(config):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def apply_lora(model, config):
    model = prepare_model_for_kbit_training(model)
    lora_cfg = config["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def get_training_args(config):
    t = config["training"]
    return TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler"],
        fp16=t["fp16"],
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        report_to="none",
    )

def main():
    config = load_config()
    print("Loading dataset...")
    dataset = get_dataset(config)
    print("Loading model...")
    model, tokenizer = load_model(config)
    model = apply_lora(model, config)
    training_args = get_training_args(config)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=config["dataset"]["max_seq_length"],
        packing=True,
    )
    print("Training...")
    trainer.train()
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])
    print("Done. Model saved to", config["output_dir"])

if __name__ == "__main__":
    main()