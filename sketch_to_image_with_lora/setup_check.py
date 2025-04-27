import torch
import importlib
import pkg_resources
import sys

required_packages = [
    "torch", "transformers", "diffusers", "accelerate",
    "peft", "Pillow", "tqdm", "safetensors"
]

def check_packages():
    print("\n📦 Checking required packages...\n")
    for pkg in required_packages:
        try:
            importlib.import_module(pkg if pkg != "Pillow" else "PIL.Image")
            version = pkg_resources.get_distribution(pkg).version
            print(f"✅ {pkg}: {version}")
        except Exception:
            print(f"❌ {pkg} is NOT properly installed!")

    # Special case for OpenCV
    try:
        import cv2
        print(f"✅ opencv-python: {cv2.__version__}")
    except ImportError:
        print("❌ opencv-python is NOT installed!")

def check_gpu():
    print("\n🖥️  Checking GPU availability...\n")
    if torch.cuda.is_available():
        print(f"✅ CUDA is available. Using: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA is NOT available. Training will fallback to CPU.")

if __name__ == "__main__":
    check_packages()
    check_gpu()
