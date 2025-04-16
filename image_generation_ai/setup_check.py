import torch
import torchvision
import os

def main():
    print("🔧 Environment Check\n" + "-"*30)
    print("PyTorch version:", torch.__version__)
    print("TorchVision version:", torchvision.__version__)
    
    print("\n💻 CUDA Info\n" + "-"*30)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Current device index:", torch.cuda.current_device())
        print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        print("CUDA capability:", torch.cuda.get_device_capability(torch.cuda.current_device()))
        print("Memory allocated (MB):", round(torch.cuda.memory_allocated() / 1024 ** 2, 2))
        print("Memory reserved  (MB):", round(torch.cuda.memory_reserved() / 1024 ** 2, 2))
    else:
        print("CUDA not available. Make sure GPU drivers and CUDA toolkit are installed.")

if __name__ == "__main__":
    main()
