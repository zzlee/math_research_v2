import torch

def inspect_model(path):
    try:
        state_dict = torch.load(path, map_location='cpu')
        print(f"\n--- Keys for {path} ---")
        for k in sorted(state_dict.keys()):
            print(f"{k}: {state_dict[k].shape}")
        return state_dict
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

v2_path = "/home/zzlee/models/mini_transformer_math_v2.pt"
v3_path = "/home/zzlee/models/mini_transformer_math_v3.pt"

inspect_model(v2_path)
inspect_model(v3_path)
