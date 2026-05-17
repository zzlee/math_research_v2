import torch

weights_path = "/home/zzlee/models/mini_transformer_math_v3.pt"
state_dict = torch.load(weights_path, map_location='cpu')

for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")
    else:
        print(f"{k}: {type(v)}")
