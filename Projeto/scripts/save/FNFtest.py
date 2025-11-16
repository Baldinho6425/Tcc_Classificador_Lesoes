import torch
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("Versão CUDA:", torch.version.cuda)
print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma GPU detectada")
