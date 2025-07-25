#!/bin/bash
apt-get update && apt-get install -y build-essential git curl
pip install --upgrade pip
pip install transformers accelerate datasets bitsandbytes
pip install nltk
pip install git+https://github.com/TransformerLensOrg/TransformerLens
pip install sae-lens
