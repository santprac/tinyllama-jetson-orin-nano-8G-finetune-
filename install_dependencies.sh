#!/bin/bash
# Quick installation script for TinyLlama fine-tuning on Jetson Orin Nano
# CUDA 12.6, JetPack 6.4.7

set -e  # Exit on error

echo "========================================================================"
echo "TinyLlama Fine-tuning Setup for Jetson Orin Nano 8GB"
echo "========================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo -e "${YELLOW}Warning: This doesn't appear to be a Jetson device${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
echo -e "${GREEN}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA
echo -e "${GREEN}Checking CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    nvcc --version | grep "release"
else
    echo -e "${RED}CUDA not found!${NC}"
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 not found! Installing...${NC}"
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip3 install --upgrade pip setuptools wheel

# Ask user which installation method
echo ""
echo "Choose installation method:"
echo "1) Upgrade existing packages (recommended if you have some packages installed)"
echo "2) Fresh install from requirements.txt"
echo "3) Minimal install (only essential packages)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo -e "${GREEN}Upgrading existing packages...${NC}"
        pip3 install --upgrade transformers==4.47.1
        pip3 install --upgrade peft==0.14.0
        pip3 install --upgrade trl==0.12.2
        pip3 install --upgrade bitsandbytes==0.43.1
        pip3 install --upgrade accelerate==1.2.1
        pip3 install --upgrade datasets==3.2.0
        pip3 install --upgrade sentencepiece protobuf scipy
        ;;
    2)
        echo -e "${GREEN}Installing from requirements.txt...${NC}"
        if [ -f requirements.txt ]; then
            pip3 install -r requirements.txt
        else
            echo -e "${RED}requirements.txt not found!${NC}"
            exit 1
        fi
        ;;
    3)
        echo -e "${GREEN}Installing minimal packages...${NC}"
        pip3 install transformers==4.47.1
        pip3 install peft==0.14.0
        pip3 install trl==0.12.2
        pip3 install bitsandbytes==0.43.1
        pip3 install accelerate==1.2.1
        pip3 install datasets==3.2.0
        ;;
    *)
        echo -e "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac

# Set up environment variables
echo ""
echo -e "${GREEN}Setting up environment variables...${NC}"
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA environment for Jetson" >> ~/.bashrc
    echo "export CUDA_HOME=/usr/local/cuda-12.6" >> ~/.bashrc
    echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    echo "" >> ~/.bashrc
    echo "# PyTorch/Transformers optimization" >> ~/.bashrc
    echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128" >> ~/.bashrc
    echo "export TOKENIZERS_PARALLELISM=false" >> ~/.bashrc
    echo "export HF_HOME=\$HOME/.cache/huggingface" >> ~/.bashrc
    echo "" >> ~/.bashrc
    echo -e "${GREEN}Environment variables added to ~/.bashrc${NC}"
    echo -e "${YELLOW}Run 'source ~/.bashrc' to apply them${NC}"
else
    echo -e "${YELLOW}CUDA environment variables already set${NC}"
fi

# Create cache directories
echo ""
echo -e "${GREEN}Creating cache directories...${NC}"
mkdir -p ~/.cache/huggingface/transformers
mkdir -p ~/.cache/huggingface/datasets

# Test installation
echo ""
echo -e "${GREEN}Running verification tests...${NC}"
echo ""

if [ -f verify_installation.py ]; then
    python3 verify_installation.py
else
    echo -e "${YELLOW}verify_installation.py not found, running basic tests...${NC}"
    
    # Basic import tests
    python3 << EOF
import sys
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
    
    import peft
    print(f"✅ PEFT: {peft.__version__}")
    
    import trl
    print(f"✅ TRL: {trl.__version__}")
    
    print("\n✅ Basic installation successful!")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF
fi

# Print next steps
echo ""
echo "========================================================================"
echo -e "${GREEN}Installation Complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "1. Source the environment: source ~/.bashrc"
echo "2. Set max performance mode: sudo nvpmodel -m 0 && sudo jetson_clocks"
echo "3. Review configuration: cat JETSON_COMPATIBILITY_GUIDE.md"
echo "4. Prepare your training data in JSON format"
echo "5. Run training: python3 train_tinyllama_jetson.py"
echo ""
echo "Helpful commands:"
echo "  - Monitor GPU: watch -n 1 tegrastats"
echo "  - Check memory: nvidia-smi"
echo "  - Test installation: python3 verify_installation.py"
echo ""
echo "========================================================================"
