
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model.")
    parser.add_argument("--base_model_path", type=str, default="./models/mistral-7b-instruct", help="Path to the base model.")
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the LoRA adapter.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model.")
    args = parser.parse_args()

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.peft_model_path)

    print("Merging model...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_path}...")
    model.save_pretrained(
        args.output_path,
        safe_serialization=True,
        max_shard_size='4GB',
    )

    # Add architectures to config
    import json
    config_path = f"{args.output_path}/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    config["architectures"] = ["MistralForCausalLM"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.peft_model_path)
    tokenizer.save_pretrained(args.output_path)

    print("Done!")

if __name__ == "__main__":
    main()
