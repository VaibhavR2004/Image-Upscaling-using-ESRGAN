import torch
model_path = "D:\Research\ESRGAN-master\models\RRDB_ESRGAN_x4.pth"
# checkpoint = torch.load(model_path, map_location="cpu")
# print(checkpoint.keys())
torch.save(model.state_dict(), "model.pth")
