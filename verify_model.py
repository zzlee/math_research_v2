import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import re

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_MODEL = 256
N_HEAD = 8  # Derived from 768 / 3 (Q,K,V) = 256, 256 / N_HEAD = 32. Let's assume N_HEAD=8.
# Wait, in_proj_weight is [768, 256]. 768 = 3 * 256. 
# The model is using a different architecture than train_math_cot.py.
# Let's redefine based on the state_dict.

BLOCK_SIZE = 64
VOCAB_SIZE = 15

DIGITS = "0123456789"
SYMBOLS = "+=|"
SPECIAL_TOKENS = ["<SOS>", "<EOS>", " "]
ALL_TOKENS = list(DIGITS) + list(SYMBOLS) + SPECIAL_TOKENS
stoi = { token: i for i, token in enumerate(ALL_TOKENS) }
itos = { i: token for i, token in enumerate(ALL_TOKENS) }

def encode(s):
    res = []
    i = 0
    while i < len(s):
        if s[i:i+5] == "<EOS>":
            res.append(stoi["<EOS>"]); i += 5
        elif s[i:i+4] == "<SOS>":
            res.append(stoi["<SOS>"]); i += 4
        elif s[i] in stoi:
            res.append(stoi[s[i]]); i += 1
        else:
            i += 1
    return res

def decode(l):
    return "".join([itos[i] for i in l if i in itos])

# ==========================================
# 2. Model Architecture (Matched to mini_transformer_math_v3.pt)
# ==========================================
class MHA(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.in_proj(x) # [B, T, 3*C]
        q, k, v = torch.split(qkv, self.d_model, dim=-1)
        
        # Simple multi-head split
        # Based on weights, it's 256. Let's assume 8 heads of 32.
        n_head = 8
        head_dim = C // n_head
        
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (head_dim**-0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class TransformerLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mha = MHA(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        # To handle the causal_mask in state_dict, we might need to store it.
        # But we can just generate it.

    def forward(self, x, mask=None):
        x = x + self.mha(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x

class MathTransformerV3(nn.Module):
    def __init__(self, vocab_size=15, d_model=256, n_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, d_model))
        self.layers = nn.ModuleList([TransformerLayer(d_model) for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embedding(idx)
        x = x + self.pos_embedding[:, :T, :]
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=DEVICE)).view(1, 1, T, T)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        logits = self.fc_out(x)
        return logits

# ==========================================
# 3. Verification Logic
# ==========================================
def generate_result(model, text, max_new_tokens=64):
    model.eval()
    idx = torch.tensor([encode(text)], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        logits = model(idx[:, -64:])
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        if idx_next.item() == stoi.get("<EOS>", -1):
            break
    return decode(idx[0].tolist())

def extract_result(generated_text):
    match = re.search(r'result=([01\d]+)', generated_text)
    if match:
        return match.group(1)
    return None

def verify_addition(model, num_tests=50):
    print("\n--- Verifying Addition ---")
    success = 0
    examples = []
    test_sets = [(1, 9, "1-digit"), (10, 99, "2-digit"), (100, 999, "3-digit"), (1000, 1999, "OOD")]
    total_tests = 0
    for low, high, label in test_sets:
        print(f"Testing {label} additions...")
        set_success = 0
        for _ in range(num_tests):
            a, b = random.randint(low, high), random.randint(low, high)
            input_text = f"{str(a)[::-1]}+{str(b)[::-1]}="
            generated = generate_result(model, input_text)
            pred_res = extract_result(generated)
            actual_res = str(a + b)
            if pred_res == actual_res: set_success += 1
            if len(examples) < 10:
                examples.append({"type": label, "input": f"{a}+{b}", "gen": generated, "actual": actual_res, "pred": pred_res, "correct": pred_res == actual_res})
            total_tests += 1
        print(f"Accuracy for {label}: {set_success/num_tests:.2%}")
        success += set_success
    print(f"Overall Addition Accuracy: {success/total_tests:.2%}")
    return examples

def verify_binary(model, num_tests=50):
    print("\n--- Verifying Binary Conversion ---")
    success = 0
    examples = []
    test_sets = [(1, 15, "Small"), (16, 100, "Medium"), (101, 200, "OOD")]
    total_tests = 0
    for low, high, label in test_sets:
        print(f"Testing {label} binary conversion...")
        set_success = 0
        for _ in range(num_tests):
            n = random.randint(low, high)
            input_text = f"{n}="
            generated = generate_result(model, input_text)
            pred_res = extract_result(generated)
            actual_res = bin(n)[2:][::-1]
            if pred_res == actual_res: set_success += 1
            if len(examples) < 10:
                examples.append({"type": label, "input": str(n), "gen": generated, "actual": actual_res, "pred": pred_res, "correct": pred_res == actual_res})
            total_tests += 1
        print(f"Accuracy for {label}: {set_success/num_tests:.2%}")
        success += set_success
    print(f"Overall Binary Accuracy: {success/total_tests:.2%}")
    return examples

if __name__ == "__main__":
    weights_path = "/home/zzlee/models/mini_transformer_math_v3.pt"
    model = MathTransformerV3().to(DEVICE)
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    # Manual load to handle naming differences and masks
    model.embedding.weight.data = state_dict['embedding.weight']
    model.pos_embedding.data = state_dict['pos_embedding']
    model.fc_out.weight.data = state_dict['fc_out.weight']
    model.fc_out.bias.data = state_dict['fc_out.bias']
    
    for i in range(4):
        layer = model.layers[i]
        layer.mha.in_proj.weight.data = state_dict[f'layers.{i}.mha.in_proj_weight']
        layer.mha.in_proj.bias.data = state_dict[f'layers.{i}.mha.in_proj_bias']
        layer.mha.out_proj.weight.data = state_dict[f'layers.{i}.mha.out_proj.weight']
        layer.mha.out_proj.bias.data = state_dict[f'layers.{i}.mha.out_proj.bias']
        layer.norm1.weight.data = state_dict[f'layers.{i}.norm1.weight']
        layer.norm1.bias.data = state_dict[f'layers.{i}.norm1.bias']
        layer.norm2.weight.data = state_dict[f'layers.{i}.norm2.weight']
        layer.norm2.bias.data = state_dict[f'layers.{i}.norm2.bias']
        layer.ff[0].weight.data = state_dict[f'layers.{i}.ff.0.weight']
        layer.ff[0].bias.data = state_dict[f'layers.{i}.ff.0.bias']
        layer.ff[2].weight.data = state_dict[f'layers.{i}.ff.2.weight']
        layer.ff[2].bias.data = state_dict[f'layers.{i}.ff.2.bias']

    print("Weights loaded successfully.")
    add_ex = verify_addition(model)
    bin_ex = verify_binary(model)
    
    print("\n--- Examples ---")
    for ex in add_ex + bin_ex:
        status = "✅" if ex["correct"] else "❌"
        print(f"{status} {ex['type']} | {ex['input']} | Pred: {ex['pred']} | Actual: {ex['actual']} | Gen: {ex['gen']}")
