import torch
import os
import ctypes
print(f'torch.__version__: {torch.__version__}')
print(f'torch.version.cuda: {torch.version.cuda}')
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
print(f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES")}')
if hasattr(torch.backends, "cuda"):
    print(f'torch.backends.cuda.is_built(): {torch.backends.cuda.is_built()}')
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    try:
        print(f'torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}')
    except Exception as e:
        print(f'Error getting device name: {e}')
else:
    print('No CUDA device available for torch')

print("\n--- Testing nvcuda.dll ---")
try:
    ctypes.WinDLL("nvcuda.dll")
    print("Success: nvcuda.dll loaded via ctypes.")
except Exception as e:
    print(f"Error: Could not load nvcuda.dll: {e}")
