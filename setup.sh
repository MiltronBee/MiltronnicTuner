#!/bin/bash

set -e

echo "🚀 Setting up Mistral-7B Fine-tuning Environment"
echo "==============================================="

VENV_NAME="mistral_finetune"
DATA_SOURCE="toor@20.163.60.124:/home/toor/copy/formatted_data.jsonl"
DATA_FILE="training_data.jsonl"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
LOG_FILE="logs/training.log"

echo "🔧 Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev build-essential libaio-dev

echo "🔧 Installing CUDA Toolkit 12.4..."
wget http://security.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.3-2ubuntu0.1_amd64.deb
sudo dpkg -i libtinfo5_6.3-2ubuntu0.1_amd64.deb
rm libtinfo5_6.3-2ubuntu0.1_amd64.deb
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
rm cuda-keyring_1.1-1_all.deb
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "✅ CUDA Toolkit installed and environment variables set."

echo "📦 Creating virtual environment..."
python3 -m venv $VENV_NAME
if [ ! -f "$VENV_NAME/bin/activate" ]; then
    echo "❌ Virtual environment creation failed."
    exit 1
fi
source $VENV_NAME/bin/activate

echo "⚡ Installing dependencies..."
$VENV_NAME/bin/python -m pip install --upgrade pip
$VENV_NAME/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$VENV_NAME/bin/python -m pip install transformers datasets peft accelerate bitsandbytes
$VENV_NAME/bin/python -m pip install wandb tqdm huggingface_hub

echo "📁 Creating directories..."
mkdir -p data models logs

echo "🔐 Setting up Hugging Face authentication..."
# Load existing .env if it exists
if [ -f ".env" ]; then
    source .env
fi

if [ -n "$HF_TOKEN" ]; then
    echo "Using existing HF_TOKEN"
    $VENV_NAME/bin/python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"
else
    echo "🔑 Hugging Face token required for Mistral-7B-Instruct access"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Also request access to: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
    echo ""
    read -p "Enter your Hugging Face token: " HF_TOKEN
    
    # Save to .env file
    echo "HF_TOKEN=$HF_TOKEN" > .env
    echo "✅ Token saved to .env file"
    
    $VENV_NAME/bin/python -m huggingface_hub.commands.huggingface_cli login --token "$HF_TOKEN"
fi

echo "📥 Checking training data..."
if [ -f "./data/$DATA_FILE" ]; then
    echo "✅ Training data already exists: ./data/$DATA_FILE"
    echo "File size: $(du -h ./data/$DATA_FILE | cut -f1)"
else
    echo "📥 Downloading training data from $DATA_SOURCE..."
    scp $DATA_SOURCE ./data/$DATA_FILE
    echo "✅ Training data downloaded: ./data/$DATA_FILE"
fi

echo "🤖 Checking/downloading base model..."
if [ -d "./models/mistral-7b-instruct" ] && [ -f "./models/mistral-7b-instruct/config.json" ] && [ -f "./models/mistral-7b-instruct/pytorch_model.bin" -o -f "./models/mistral-7b-instruct/pytorch_model-00001-of-00002.bin" ]; then
    echo "✅ Model already exists: ./models/mistral-7b-instruct"
    echo "Model size: $(du -sh ./models/mistral-7b-instruct | cut -f1)"
else
    echo "📥 Downloading Mistral-7B-Instruct model..."
    $VENV_NAME/bin/python -c "
from huggingface_hub import snapshot_download
import os

print('Downloading Mistral-7B-Instruct model...')
snapshot_download(
    repo_id='$MODEL_NAME',
    local_dir='./models/mistral-7b-instruct',
    local_dir_use_symlinks=False
)
print('✅ Model downloaded and saved!')
"
fi

echo ""
echo "🔐 Setting up wandb..."
if [ -n "$WANDB_API_KEY" ]; then
    echo "Using existing WANDB_API_KEY"
    $VENV_NAME/bin/python -m wandb.cli.cli login --relogin "$WANDB__API_KEY"
else
    echo "📊 WandB API key required for training monitoring"
    echo "Get your key from: https://wandb.ai/authorize"
    echo ""
    read -p "Enter your WandB API key: " WANDB_API_KEY
    
    # Append to .env file
    echo "WANDB_API_KEY=$WANDB_API_KEY" >> .env
    echo "✅ WandB key saved to .env file"
    
    $VENV_NAME/bin/python -m wandb.cli.cli login --relogin "$WANDB_API_KEY"
fi
echo "✅ WandB authentication complete"

echo "🏃 Starting training..."
echo "📝 Check logs in the console and at https://wandb.ai"

$VENV_NAME/bin/python -m accelerate.commands.accelerate_cli launch train.py \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--num_train_epochs 3 \
--logging_steps 10 \
--save_steps 500 \
--max_seq_length 2048 \
--num_workers 16 \
--report_to wandb
