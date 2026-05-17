import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================\
# 1. Configuration & Tokenizer\
# ==========================================\
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_MODEL = 128
N_LAYER = 3
N_HEAD = 4
VOCAB_SIZE = 15
BLOCK_SIZE = 64

# Based on vocab_size=15: 10 digits + 3 symbols (+, =, |) + 2 special (<SOS>, <EOS>)
ALL_TOKENS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", "|", "<SOS>", "<EOS>"]
stoi = {token: i for i, token in enumerate(ALL_TOKENS)}
itos = {i: token for i, token in enumerate(ALL_TOKENS)}

def encode(s):
    res = []
    i = 0
    while i < len(s):
        if s[i:i+5] == "<EOS>":
            res.append(stoi["<EOS>"]); i += 5
        elif s[i:i+5] == "<SOS>":
            res.append(stoi["<SOS>"]); i += 5
        elif s[i] in stoi:
            res.append(stoi[s[i]]); i += 1
        else:
            i += 1
    return res

def decode(l):
    return "".join([itos[i] for i in l if i in itos])

# ==========================================\
# 2. Model Architecture (Matches V2 Weights)\
# ==========================================\
class ModelV2(nn.Module):
    def __init__(self, vocab_size=15, d_model=128, n_layers=3, n_head=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 16, d_model)) 
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mha': nn.MultiheadAttention(d_model, n_head, batch_first=True),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU(),
                    nn.Linear(4 * d_model, d_model)
                )
            }) for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embedding(idx)
        # pos_embedding is [1, 64, d_model] or [1, 16, d_model]
        # We take the slice needed.
        pos = self.pos_embedding[:, :T, :]
        x = x + pos
        
        # Causal mask for MHA
        mask = torch.full((T, T), float('-inf'), device=DEVICE)
        mask = torch.triu(mask, diagonal=1)
        # To match MultiheadAttention's expected mask (0 for keep, -inf for mask)
        # But wait, torch.triu(mask, diagonal=1) already puts -inf in upper tri.
        # We need the lower tri to be 0.
        mask = torch.where(mask == float('-inf'), float('-inf'), 0.0)
        # The above is actually: mask[i, j] = -inf if j > i else 0.
        # Actually, simpler:
        # mask = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
        # attn_mask = torch.zeros(T, T, device=DEVICE).masked_fill(mask, float('-inf'))
        
        for layer in self.layers:
            norm_x = layer['norm1'](x)
            # Apply causal mask
            attn_mask = torch.triu(torch.ones(T, T, device=DEVICE), diagonal=1).bool()
            attn_mask = torch.zeros(T, T, device=DEVICE).masked_fill(attn_mask, float('-inf'))
            
            attn_out, _ = layer['mha'](norm_x, norm_x, norm_x, attn_mask=attn_mask)
            x = x + attn_out
            x = x + layer['ff'](layer['norm2'](x))
            
        logits = self.fc_out(x)
        return logits

def load_model_v2(path):
    model = ModelV2().to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    
    # The weights in the file might have slightly different names or structure
    # We need to map state_dict keys to model parameters
    new_state_dict = {}
    
    # embedding.weight -> embedding.weight
    new_state_dict['embedding.weight'] = state_dict['embedding.weight']
    # pos_embedding -> pos_embedding
    new_state_dict['pos_embedding'] = state_dict['pos_embedding']
    
    for i in range(N_LAYER):
        # layers.i.mha.in_proj_weight -> layers[i].mha.in_proj_weight
        new_state_dict[f'layers.{i}.mha.in_proj_weight'] = state_dict[f'layers.{i}.mha.in_proj_weight']
        new_state_dict[f'layers.{i}.mha.in_proj_bias'] = state_dict[f'layers.{i}.mha.in_proj_bias']
        new_state_dict[f'layers.{i}.mha.out_proj.weight'] = state_dict[f'layers.{i}.mha.out_proj.weight']
        new_state_dict[f'layers.{i}.mha.out_proj.bias'] = state_dict[f'layers.{i}.mha.out_proj.bias']
        new_state_dict[f'layers.{i}.norm1.weight'] = state_dict[f'layers.{i}.norm1.weight']
        new_state_dict[f'layers.{i}.norm1.bias'] = state_dict[f'layers.{i}.norm1.bias']
        new_state_dict[f'layers.{i}.norm2.weight'] = state_dict[f'layers.{i}.norm2.weight']
        new_state_dict[f'layers.{i}.norm2.bias'] = state_dict[f'layers.{i}.norm2.bias']
        new_state_dict[f'layers.{i}.ff.0.weight'] = state_dict[f'layers.{i}.ff.0.weight']
        new_state_dict[f'layers.{i}.ff.0.bias'] = state_dict[f'layers.{i}.ff.0.bias']
        new_state_dict[f'layers.{i}.ff.2.weight'] = state_dict[f'layers.{i}.ff.2.weight']
        new_state_dict[f'layers.{i}.ff.2.bias'] = state_dict[f'layers.{i}.ff.2.bias']
        
    new_state_dict['fc_out.weight'] = state_dict['fc_out.weight']
    new_state_dict['fc_out.bias'] = state_dict['fc_out.bias']
    
    # We need to handle the ModuleDict/ModuleList structure
    # The easiest way is to just load the state_dict and let torch handle the naming if we name our modules correctly.
    # But we used a slightly different structure. Let's fix it.
    
    # Actually, if I rename the modules in ModelV2 to match the keys:
    # self.layers = nn.ModuleList([
    #     nn.ModuleDict({ ... })
    # ])
    # The keys will be layers.0.mha... which matches!
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def generate_result(model, text, max_new_tokens=64):
    idx = torch.tensor([encode(text)], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -16:])
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next.item() == stoi.get("<EOS>", -1):
            break
    return decode(idx[0].tolist())

# ==========================================\
# 3. Test Suite\
# ==========================================\
def run_tests():
    model = load_model_v2('/home/zzlee/models/mini_transformer_math_v2.pt')
    
    test_cases = []
    
    # Addition (Input reversed)
    # Format: "s_a+s_b= " where s_a, s_b are reversed
    # 1-digit
    for a in [1, 5, 9]:
        for b in [1, 5, 9]:
            test_cases.append(('add', a, b, f"{str(a)[::-1]}+{str(b)[::-1]}="))
    # 2-digit
    for a in [10, 25, 99]:
        for b in [10, 25, 99]:
            test_cases.append(('add', a, b, f"{str(a)[::-1]}+{str(b)[::-1]}="))
    # 3-digit
    for a in [100, 250, 999]:
        for b in [100, 250, 999]:
            test_cases.append(('add', a, b, f"{str(a)[::-1]}+{str(b)[::-1]}="))
    # Different lengths
    test_cases.append(('add', 1, 10, "1+01="))
    test_cases.append(('add', 10, 100, "01+001="))
    # OOD
    test_cases.append(('add', 1000, 1000, "0001+0001="))

    # Binary Conversion
    # Format: "n= "
    for n in [1, 2, 5, 10, 15, 32, 64, 100]:
        test_cases.append(('bin', n, None, f"{n}="))
    # OOD
    test_cases.append(('bin', 255, None, "255="))

    results = []
    for task, val1, val2, prompt in test_cases:
        generated = generate_result(model, prompt)
        
        if task == 'add':
            expected_res = str(val1 + val2)
            # Result is usually at the end after "result="
            # CoT: "2+1=3|...result=123"
            actual_res = generated.split('result=')[-1].strip()
            correct = (actual_res == expected_res)
        else:
            expected_res = bin(val1)[2:]
            actual_res = generated.split('result=')[-1].strip()
            correct = (actual_res == expected_res)
            
        results.append((task, prompt, expected_res, actual_res, correct))
        
    return results

if __name__ == "__main__":
    res = run_tests()
    
    print(f"{'Task':<10} | {'Prompt':<15} | {'Expected':<10} | {'Actual':<10} | {'Correct'}")
    print("-" * 60)
    for r in res:
        print(f"{r[0]:<10} | {r[1]:<15} | {r[2]:<10} | {r[3]:<10} | {r[4]}")
    
    add_res = [r for r in res if r[0] == 'add']
    bin_res = [r for r in res if r[0] == 'bin']
    
    print(f"\nAddition Accuracy: {sum(r[4] for r in add_res)/len(add_res):.2%}")
    print(f"Binary Accuracy: {sum(r[4] for r in bin_res)/len(bin_res):.2%}")
    print(f"Overall Accuracy: {sum(r[4] for r in res)/len(res):.2%}")
