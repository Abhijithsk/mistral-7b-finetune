from transformers import pipeline
import torch

def load_pipeline(model_path="./mistral-7b-merged"):
    return pipeline(
        "text-generation",
        model=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

def ask(pipe, instruction, max_new_tokens=256):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    out = pipe(prompt, max_new_tokens=max_new_tokens,
               do_sample=True, temperature=0.7, top_p=0.9)
    return out[0]["generated_text"].split("### Response:\n")[-1].strip()

if __name__ == "__main__":
    pipe = load_pipeline()
    while True:
        q = input("\nInstruction (or 'quit'): ")
        if q.lower() == "quit":
            break
        print("\nResponse:", ask(pipe, q))