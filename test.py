import torch
print("Versi PyTorch:", torch.__version__)
print("Built with CUDA:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Detected:", torch.cuda.get_device_name(0))
else:
    print("GPU tidak terdeteksi.")