from datasets import load_dataset
import yaml

def load_config(path="configs/lora_config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def format_medical_sample(example):
    return {
        "text": (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    }

def get_dataset(config):
    cfg = config["dataset"]
    dataset = load_dataset(cfg["name"], split=f"train[:{cfg['subset_size']}]")
    dataset = dataset.map(format_medical_sample, remove_columns=dataset.column_names)
    return dataset.train_test_split(test_size=cfg["test_size"], seed=42)