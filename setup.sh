#!/bin/bash
set -e

# System deps
apt-get update && apt-get install -y build-essential git curl

# Upgrade pip
pip install --upgrade pip

# Core Hugging Face stack (pin transformers to 4.46.3 for DeepSeek compatibility)
pip install "transformers==4.46.3" accelerate datasets safetensors

# Quantization (CUDA 12.1 supported)
pip install bitsandbytes

# TorchVision matching PyTorch 2.3.1 + CUDA 12.1
pip install torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# NLP utils
pip install nltk

# Research tools
pip install git+https://github.com/TransformerLensOrg/TransformerLens
pip install sae-lens
