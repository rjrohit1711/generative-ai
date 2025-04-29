import importlib
import torch

def check_torch():
    print("Torch Version:", torch.__version__)
    if torch.cuda.is_available():
        print(f"✅ GPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ GPU not detected.")

def check_modules(modules):
    for module in modules:
        spec = importlib.util.find_spec(module)
        if spec is None:
            print(f"❌ Module missing: {module}")
        else:
            print(f"✅ Module found: {module}")

def main():
    print("🔎 Checking environment setup...\n")
    check_torch()
    print("\n🔎 Checking required Python packages...\n")
    check_modules([
        "langchain",
        "huggingface_hub",
        "diffusers",
        "transformers",
        "audiocraft",
        "dotenv",
        "accelerate"
    ])

if __name__ == "__main__":
    main()
