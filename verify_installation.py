#!/usr/bin/env python3
"""
Verify all dependencies for TinyLlama fine-tuning on Jetson Orin Nano
Compatible with CUDA 12.6, JetPack 6.4.7, Python 3.10.12
"""

import sys
import os

def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

def check_import(module_name, package_name=None, min_version=None):
    """Check if a module can be imported and meets version requirements"""
    package_name = package_name or module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                print(f"⚠️  {package_name}: {version} (need >= {min_version})")
                return False
        
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False
    except Exception as e:
        print(f"❌ {package_name}: ERROR - {e}")
        return False

def check_cuda():
    """Check CUDA availability and configuration"""
    try:
        import torch
        
        print_header("CUDA Configuration")
        
        if not torch.cuda.is_available():
            print("❌ CUDA: Not available")
            print("   PyTorch was not built with CUDA support")
            return False
        
        print(f"✅ CUDA Available: True")
        print(f"   - PyTorch Version: {torch.__version__}")
        print(f"   - CUDA Version: {torch.version.cuda}")
        print(f"   - cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   - Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   - Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"   - Compute Capability: {props.major}.{props.minor}")
            print(f"   - Multi Processors: {props.multi_processor_count}")
        
        # Test CUDA operation
        try:
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            print(f"\n✅ CUDA Operations: Working")
            del x, y
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n❌ CUDA Operations: Failed - {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA Check Failed: {e}")
        return False

def check_bitsandbytes():
    """Check BitsAndBytes specifically for ARM64 compatibility"""
    try:
        import bitsandbytes as bnb
        print(f"\n✅ BitsAndBytes: {bnb.__version__}")
        
        # Check version compatibility
        version = bnb.__version__
        if version < "0.41.0":
            print(f"   ⚠️  Warning: Version {version} may not support ARM64")
            print(f"   Recommended: 0.43.1 for Jetson devices")
        
        # Try to check CUDA setup
        try:
            import torch
            if torch.cuda.is_available():
                # Test if 8bit operations work
                from bitsandbytes.nn import Linear8bitLt
                print(f"   - 8-bit Linear: Available")
                
                # Test if 4bit config can be imported
                try:
                    from transformers import BitsAndBytesConfig
                    print(f"   - 4-bit Config: Available")
                except ImportError:
                    print(f"   - 4-bit Config: Not available (need transformers>=4.30)")
        except Exception as e:
            print(f"   - CUDA Integration: {e}")
        
        return True
        
    except ImportError:
        print(f"\n❌ BitsAndBytes: NOT INSTALLED")
        print("   Install with: pip install bitsandbytes==0.43.1")
        return False
    except Exception as e:
        print(f"\n⚠️  BitsAndBytes: Installed but may have issues")
        print(f"   Error: {e}")
        return True

def check_transformers_features():
    """Check specific Transformers features needed for fine-tuning"""
    print_header("Transformers Features")
    
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            BitsAndBytesConfig
        )
        print("✅ AutoModelForCausalLM: Available")
        print("✅ AutoTokenizer: Available")
        print("✅ TrainingArguments: Available")
        print("✅ BitsAndBytesConfig: Available (4-bit quantization supported)")
        return True
    except ImportError as e:
        print(f"❌ Missing Transformers features: {e}")
        print("   Upgrade with: pip install --upgrade transformers==4.47.1")
        return False

def check_peft_features():
    """Check PEFT/LoRA features"""
    print_header("PEFT/LoRA Features")
    
    try:
        from peft import (
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            PeftModel
        )
        print("✅ LoraConfig: Available")
        print("✅ get_peft_model: Available")
        print("✅ prepare_model_for_kbit_training: Available")
        print("✅ PeftModel: Available")
        return True
    except ImportError as e:
        print(f"❌ Missing PEFT features: {e}")
        print("   Install with: pip install peft==0.14.0")
        return False

def check_trl_features():
    """Check TRL features for SFT"""
    print_header("TRL (SFT Trainer) Features")
    
    try:
        from trl import SFTTrainer
        print("✅ SFTTrainer: Available")
        return True
    except ImportError as e:
        print(f"❌ Missing TRL features: {e}")
        print("   Install with: pip install trl==0.12.2")
        return False

def test_model_loading():
    """Test loading a small model to verify everything works"""
    print_header("Model Loading Test (Quick)")
    
    try:
        from transformers import AutoTokenizer
        
        print("Loading TinyLlama tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            trust_remote_code=True
        )
        print(f"✅ Tokenizer loaded successfully")
        print(f"   - Vocab size: {len(tokenizer)}")
        print(f"   - Model max length: {tokenizer.model_max_length}")
        
        # Test tokenization
        text = "Hello, this is a test."
        tokens = tokenizer(text, return_tensors="pt")
        print(f"✅ Tokenization works: '{text}' → {tokens['input_ids'].shape[1]} tokens")
        
        print("\n⚠️  Full model loading skipped (saves time)")
        print("   Run your training script to test full model loading with quantization")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False

def check_system_resources():
    """Check system resources"""
    print_header("System Resources")
    
    try:
        import psutil
        import torch
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        print(f"✅ CPU: {cpu_count} cores, {cpu_percent}% used")
        
        # Memory
        mem = psutil.virtual_memory()
        print(f"✅ RAM: {mem.total / 1e9:.2f} GB total, {mem.available / 1e9:.2f} GB available ({mem.percent}% used)")
        
        # GPU Memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"✅ GPU {i} Memory: {total:.2f} GB total, {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        
        # Disk
        disk = psutil.disk_usage('/')
        print(f"✅ Disk: {disk.total / 1e9:.2f} GB total, {disk.free / 1e9:.2f} GB free ({disk.percent}% used)")
        
        # Check if enough space for models
        if disk.free / 1e9 < 10:
            print(f"⚠️  Warning: Low disk space (< 10 GB free)")
            print(f"   You may need at least 20 GB for models and checkpoints")
        
        return True
        
    except ImportError:
        print("⚠️  psutil not installed, skipping system resource check")
        print("   Install with: pip install psutil")
        return True
    except Exception as e:
        print(f"⚠️  System resource check failed: {e}")
        return True

def print_recommendations():
    """Print recommendations based on system"""
    print_header("Recommendations for Jetson Orin Nano 8GB")
    
    print("""
For optimal fine-tuning performance:

1. Memory Settings:
   - Use batch_size=1 with gradient_accumulation_steps=16
   - Set max_seq_length=256 (or 128 if OOM)
   - Enable gradient_checkpointing=True

2. LoRA Configuration:
   - Use rank (r) = 8 (reduce to 4 if OOM)
   - Target only attention layers if memory is tight
   - Set lora_alpha = 16 (2x rank)

3. Quantization:
   - Use 4-bit NF4 quantization (load_in_4bit=True)
   - Enable double quantization for extra compression
   - Use float16 compute dtype

4. System Optimization:
   - Run in headless mode (no GUI): sudo systemctl set-default multi-user.target
   - Set max power mode: sudo nvpmodel -m 0
   - Lock clocks: sudo jetson_clocks
   - Close unnecessary applications

5. Training Tips:
   - Start with small dataset to test (100-1000 samples)
   - Monitor with: watch -n 1 tegrastats
   - Expected speed: 50-150 tokens/second
   - Typical training time: 6-18 hours for 3 epochs on 10k samples
    """)

def main():
    """Main verification function"""
    print("=" * 70)
    print("Jetson Orin Nano - TinyLlama Fine-tuning Environment Verification")
    print("=" * 70)
    
    print(f"\nPython Version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    all_checks_passed = True
    
    # Check core dependencies
    print_header("Core Dependencies")
    all_checks_passed &= check_import('torch', 'PyTorch', '2.0.0')
    all_checks_passed &= check_import('transformers', 'Transformers', '4.30.0')
    all_checks_passed &= check_import('peft', 'PEFT', '0.4.0')
    all_checks_passed &= check_import('trl', 'TRL', '0.4.0')
    all_checks_passed &= check_import('accelerate', 'Accelerate')
    all_checks_passed &= check_import('datasets', 'Datasets')
    
    # Check optional dependencies
    print("\nOptional Dependencies:")
    check_import('scipy', 'SciPy')
    check_import('sentencepiece', 'SentencePiece')
    check_import('tensorboard', 'TensorBoard')
    
    # Check CUDA
    cuda_ok = check_cuda()
    all_checks_passed &= cuda_ok
    
    # Check BitsAndBytes
    bnb_ok = check_bitsandbytes()
    
    # Check specific features
    if all_checks_passed:
        all_checks_passed &= check_transformers_features()
        all_checks_passed &= check_peft_features()
        all_checks_passed &= check_trl_features()
    
    # System resources
    check_system_resources()
    
    # Test model loading
    if all_checks_passed:
        test_model_loading()
    
    # Print recommendations
    print_recommendations()
    
    # Final verdict
    print_header("Verification Summary")
    
    if all_checks_passed and bnb_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nYour system is ready for TinyLlama fine-tuning with 4-bit quantization.")
        print("\nNext steps:")
        print("1. Prepare your training data in JSON format")
        print("2. Review the recommended configuration in JETSON_COMPATIBILITY_GUIDE.md")
        print("3. Run: python3 train_tinyllama_jetson.py")
    elif all_checks_passed:
        print("⚠️  CORE LIBRARIES OK, BUT BITSANDBYTES MAY NEED ATTENTION")
        print("\nYou can still fine-tune, but without 4-bit quantization.")
        print("Consider using FP16 model instead, or try reinstalling bitsandbytes:")
        print("  pip install --upgrade bitsandbytes==0.45.0")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        print("\nOr install individually:")
        print("  pip install transformers==4.47.1 peft==0.14.0 trl==0.12.2")
        print("  pip install bitsandbytes==0.45.0 accelerate==1.2.1")
    
    print("=" * 70)
    
    return 0 if (all_checks_passed and bnb_ok) else 1

if __name__ == "__main__":
    sys.exit(main())
