import torch
# print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
model = torch.load("./runs/train/exp/weights/best.pt", map_location=device)['model'].float()
model.to(device).eval()
model.half()
print(model.names)