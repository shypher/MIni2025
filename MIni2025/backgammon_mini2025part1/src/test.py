import os
import torch

# Check RLNN.pth file
file_path = "c:/Users/shay1/Documents/GitHub/MIni2025/MIni2025/backgammon_mini2025part1/RLNN.pth"
print(f"File exists: {os.path.isfile(file_path)}")

# Check CUDA
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
else:
    print("Torch is not using CUDA. Reinstall with the correct CUDA version.")
