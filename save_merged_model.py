
import torch
from unsloth import FastLanguageModel
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter and save it.")
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the LoRA adapter.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model.")
    args = parser.parse_args()

    print("Loading base model with unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        max_seq_length = 4096,
        dtype = None,
        load_in_4bit = True,
    )

    print("Merging LoRA adapter...")
    model = FastLanguageModel.from_pretrained(
        model,
        args.peft_model_path,
    )

    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    print("Done!")

if __name__ == "__main__":
    main()
