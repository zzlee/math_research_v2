import torch

def inspect_model(path):
    try:
        state_dict = torch.load(path, map_location='cpu')
        # Use the state_dict to derive architecture
        # d_model is the second dimension of the embedding weight
        d_model = state_dict['token_embedding_table.weight'].shape[1]
        
        # Layers are identified by the index in 'blocks.N...'
        block_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('blocks.') and len(k.split('.')) > 1]
        num_layers = max(block_indices) + 1 if block_indices else 0
        
        # Vocab size is the first dimension of the embedding weight
        vocab_size = state_dict['token_embedding_table.weight'].shape[0]
        
        return {
            "d_model": d_model,
            "num_layers": num_layers,
            "vocab_size": vocab_size
        }
    except Exception as e:
        return {"error": str(e)}

v2_path = "/home/zzlee/models/mini_transformer_math_v2.pt"
v3_path = "/home/zzlee/models/mini_transformer_math_v3.pt"

print(f"V2: {inspect_model(v2_path)}")
print(f"V3: {inspect_model(v3_path)}")
