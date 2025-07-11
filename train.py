#!/usr/bin/env python3
import os
import json
import sys
import torch
import wandb
import argparse
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, default_data_collator, logging as hf_logging
)
hf_logging.set_verbosity_error()
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl_data(file_path):
    """Load JSONL training data with instruct format"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            # Extract the text from the 'text' key
            text = item.get("text", "")
            
            # Find the instruction and output parts
            inst_start = text.find("[INST]") + len("[INST]")
            inst_end = text.find("[/INST]")
            
            if inst_start == -1 or inst_end == -1:
                continue

            instruction = text[inst_start:inst_end].strip()
            output = text[inst_end + len("[/INST]"):].strip()

            # The 'input' field is not present in this data format
            data.append({
                "instruction": instruction,
                "input": "", 
                "output": output
            })
    return data

def main():
    parser = argparse.ArgumentParser(description="Mistral 7B Fine-tuning")
    parser.add_argument("--model_name_or_path", type=str, default="./models/mistral-7b-instruct", help="Model name or path")
    parser.add_argument("--train_file", type=str, default="./data/training_data.jsonl", help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./models/mistral-7b-finetuned", help="Output directory for fine-tuned model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per device for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--report_to", type=str, default="wandb", help="Report to wandb or none")
    parser.add_argument("--num_workers", type=int, default=1, help="Dataloader number of workers")
    args = parser.parse_args()

    try:
        # Check available GPUs
        device_count = torch.cuda.device_count()
        logger.info(f"🎮 Available CUDA devices: {device_count}")
        
        if device_count == 0:
            logger.error("No CUDA devices found! Training requires GPU.")
            return
    
        # Initialize wandb
        if os.environ.get("LOCAL_RANK", "0") == "0" and args.report_to == "wandb":
            wandb.init(
                project="mistral-7b-finetune",
                name=f"mistral-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": args.model_name_or_path,
                    "dataset": args.train_file,
                    "method": "LoRA",
                    "gpu_count": device_count,
                    **vars(args)
                }
            )
    
        # Configuration
        model_path = args.model_name_or_path
        data_path = args.train_file
        output_dir = args.output_dir
    
        logger.info("🚀 Starting Mistral-7B fine-tuning...")
        
        # Load tokenizer and model
        logger.info("Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, # Use bfloat16 for H100
            trust_remote_code=True
        )
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
        # Load and prepare data
        logger.info("Loading training data...")
        raw_data = load_jsonl_data(data_path)
        
        # Create dataset
        dataset = Dataset.from_list(raw_data)

        if os.environ.get("LOCAL_RANK", "0") == "0" and args.report_to == "wandb":
            logger.info("📊 Logging a sample of the dataset to wandb...")
            try:
                sample_dataset = dataset.shuffle().select(range(100))
                df = sample_dataset.to_pandas()
                table = wandb.Table(dataframe=df)
                wandb.log({"training_data_sample": table})
            except Exception as e:
                logger.warning(f"Could not log dataset sample to wandb: {e}")

        def processing_function(examples, tokenizer, max_length):
            prompts = []
            texts = []
            for i in range(len(examples['instruction'])):
                instruction = examples['instruction'][i]
                input_text = examples['input'][i]
                output = examples['output'][i]

                if input_text:
                    prompt = f"[INST] {instruction}\n\n{input_text} [/INST]"
                else:
                    prompt = f"[INST] {instruction} [/INST]"
                
                texts.append(f"{prompt} {output}")
                prompts.append(prompt)

            model_inputs = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            # Clone input_ids to create labels
            labels = [list(x) for x in model_inputs["input_ids"]]
            
            # Get tokenized prompts to calculate lengths
            prompt_tokenized = tokenizer(
                prompts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            prompt_lengths = [sum(x) for x in prompt_tokenized['attention_mask']]


            # Mask out the prompt part of the labels
            for i in range(len(labels)):
                prompt_len = prompt_lengths[i]
                for j in range(prompt_len):
                    labels[i][j] = -100

            model_inputs["labels"] = labels
            return model_inputs

        tokenized_dataset = dataset.map(
            lambda examples: processing_function(examples, tokenizer, args.max_seq_length),
            batched=True,
            num_proc=16,
            load_from_cache_file=True,
            remove_columns=dataset.column_names
        )
        
        # Split dataset
        train_size = int(0.9 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size, # Use same for eval
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            learning_rate=2e-4,
            bf16=True, # Enable bfloat16 for H100
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.save_steps, # Match save_steps
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=args.report_to,
            run_name=f"mistral-finetune-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dataloader_num_workers=args.num_workers,
            remove_unused_columns=True,
            label_names=["labels"],
            dataloader_pin_memory=True, # Enable pin_memory
            group_by_length=False,
        )
    
        # Data collator
        data_collator = default_data_collator
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    
        # Start training
        logger.info("🏃 Starting training...")
        trainer.train()
        
        # Save final model
        logger.info("💾 Saving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save LoRA adapter
        model.save_pretrained(f"{output_dir}/lora_adapter")
        
        logger.info("Merging the adapter with the base model...")
        # load base model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        # load peft model
        model = PeftModel.from_pretrained(
            model,
            output_dir,
        )
        # merge models
        model = model.merge_and_unload()
        logger.info("💾 Saving final model...")
        model.save_pretrained(
            output_dir,
            safe_serialization=True, 
            max_shard_size='4GB',
        )
        tokenizer.save_pretrained(output_dir)

        logger.info("✅ Training completed successfully!")
        if os.environ.get("LOCAL_RANK", "0") == "0" and args.report_to == "wandb":
            wandb.finish()
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        if os.environ.get("LOCAL_RANK", "0") == "0" and args.report_to == "wandb":
            wandb.finish()
        raise

if __name__ == "__main__":
    main()
