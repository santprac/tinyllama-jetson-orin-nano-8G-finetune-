#!/usr/bin/env python3
"""
Quick Setup Test for Jetson Orin Nano
Tests: Transformers, CUDA, Accelerate, PEFT, TRL
"""

import sys

def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_basic_imports():
    """Test if all packages can be imported"""
    print_section("1. Testing Package Imports")
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'peft': 'PEFT',
        'trl': 'TRL',
        'accelerate': 'Accelerate',
        'datasets': 'Datasets',
        'numpy': 'NumPy',
    }
    
    all_ok = True
    for module, name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {name:20s} {version}")
        except ImportError as e:
            print(f"❌ {name:20s} NOT INSTALLED - {e}")
            all_ok = False
    
    return all_ok

def test_cuda():
    """Test CUDA availability and configuration"""
    print_section("2. Testing CUDA")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA is not available!")
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
        print("\n   Testing CUDA operations...")
        x = torch.randn(100, 100).cuda()
        y = x @ x.T
        result = y.cpu().numpy()
        print(f"✅ Matrix multiplication on GPU: Success")
        
        # Memory check
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\n   GPU Memory:")
        print(f"   - Allocated: {allocated:.3f} GB")
        print(f"   - Reserved: {reserved:.3f} GB")
        
        # Cleanup
        del x, y
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_transformers():
    """Test Transformers library and tokenizer"""
    print_section("3. Testing Transformers")
    
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
        text = "Hello, this is a test of the TinyLlama tokenizer!"
        tokens = tokenizer(text, return_tensors="pt")
        
        print(f"\n   Testing tokenization:")
        print(f"   - Input: '{text}'")
        print(f"   - Tokens: {tokens['input_ids'].shape[1]} tokens")
        print(f"   - Token IDs: {tokens['input_ids'][0][:10].tolist()}... (first 10)")
        
        # Test decoding
        decoded = tokenizer.decode(tokens['input_ids'][0])
        print(f"   - Decoded: '{decoded[:50]}...'")
        
        print(f"\n✅ Transformers working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test loading TinyLlama model in FP16"""
    print_section("4. Testing Model Loading (FP16)")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM
        
        print("Loading TinyLlama-1.1B in FP16...")
        print("(This will take 30-60 seconds, please wait...)")
        
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        memory_gb = model.get_memory_footprint() / 1e9
        print(f"\n✅ Model loaded successfully!")
        print(f"   - Model size: {memory_gb:.2f} GB")
        print(f"   - Device: {model.device}")
        print(f"   - Data type: {model.dtype}")
        
        # Check GPU memory
        allocated = torch.cuda.memory_allocated() / 1e9
        available = 8.0 - allocated
        print(f"\n   GPU Memory:")
        print(f"   - Used: {allocated:.2f} GB")
        print(f"   - Available: {available:.2f} GB")
        
        if available > 4.0:
            print(f"   ✅ Plenty of memory for fine-tuning!")
        elif available > 2.0:
            print(f"   ⚠️  Sufficient memory, but tight")
        else:
            print(f"   ❌ May not have enough memory for fine-tuning")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_peft():
    """Test PEFT/LoRA functionality"""
    print_section("5. Testing PEFT/LoRA")
    
    try:
        from peft import LoraConfig, get_peft_model
        import torch
        from torch import nn
        
        print("Testing LoRA configuration...")
        
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 100)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Create LoRA config
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["linear"],
            lora_dropout=0.05,
            bias="none",
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        print(f"✅ LoRA configuration successful!")
        model.print_trainable_parameters()
        
        return True
        
    except Exception as e:
        print(f"❌ PEFT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_accelerate():
    """Test Accelerate functionality"""
    print_section("6. Testing Accelerate")
    
    try:
        from accelerate import Accelerator
        
        print("Testing Accelerate...")
        accelerator = Accelerator()
        
        print(f"✅ Accelerate initialized successfully!")
        print(f"   - Device: {accelerator.device}")
        print(f"   - Mixed precision: {accelerator.mixed_precision}")
        print(f"   - Distributed: {accelerator.distributed_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Accelerate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trl():
    """Test TRL/SFTTrainer"""
    print_section("7. Testing TRL")
    
    try:
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        print("Testing TRL imports...")
        print(f"✅ SFTTrainer imported successfully!")
        print(f"✅ TrainingArguments imported successfully!")
        
        # Create minimal config to test
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            num_train_epochs=1,
        )
        
        print(f"✅ TrainingArguments created successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ TRL test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference():
    """Test simple inference with TinyLlama"""
    print_section("8. Testing Inference (Optional - May Take Time)")
    
    response = input("\nRun inference test? This will load the model and generate text (y/n): ")
    if response.lower() != 'y':
        print("⏭️  Skipping inference test")
        return True
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\nLoading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        )
        
        prompt = "### Instruction:\nWhat is artificial intelligence?\n\n### Response:\n"
        
        print(f"\nGenerating response for prompt:")
        print(f"'{prompt}'")
        print("\nGenerating (this may take 10-30 seconds)...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n✅ Inference successful!")
        print(f"\nGenerated response:")
        print("-" * 70)
        print(response)
        print("-" * 70)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_summary(results):
    """Print final summary"""
    print_section("SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTest Results: {passed_tests}/{total_tests} passed\n")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10s} {test_name}")
    
    print("\n" + "=" * 70)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour Jetson Orin Nano is ready for TinyLlama fine-tuning!")
        print("\nNext steps:")
        print("1. Create your training data (training_data.json)")
        print("2. Review FINAL_SETUP_SUMMARY.md for training guide")
        print("3. Set max performance: sudo nvpmodel -m 0 && sudo jetson_clocks")
        print("4. Start training!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease review the error messages above.")
        print("Check SOLUTION_NO_BITSANDBYTES.md for troubleshooting.")
    
    print("=" * 70)

def main():
    """Run all tests"""
    print("=" * 70)
    print("  Jetson Orin Nano Setup Verification")
    print("  Testing: Transformers, CUDA, Accelerate, PEFT, TRL")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results["Package Imports"] = test_basic_imports()
    results["CUDA"] = test_cuda()
    results["Transformers"] = test_transformers()
    results["Model Loading"] = test_model_loading()
    results["PEFT/LoRA"] = test_peft()
    results["Accelerate"] = test_accelerate()
    results["TRL"] = test_trl()
    results["Inference"] = test_inference()
    
    # Print summary
    print_summary(results)
    
    # Exit code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
