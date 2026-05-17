import torch
import torch.nn as nn
from torch.nn import functional as F
import random

# =========================================
# 1. Config & Tokenizer (Matching train_math_cot.py)
# =========================================
D_MODEL = 128
N_LAYER = 3
VOCAB_SIZE = 15
BLOCK_SIZE = 64
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SPECIAL_TOKENS = ["<SOS>", "<EOS>", " "]
DIGITS = "0123456789"
SYMBOLS = "+=|"
ALL_TOKENS = list(DIGITS) + list(SYMBOLS) + SPECIAL_TOKENS
stoi = { token: i for i, token in enumerate(ALL_TOKENS) }
itos = { i: token for i, token in enumerate(ALL_TOKENS) }

def encode(s):
    res = []
    i = 0
    while i < len(s):
        if s[i:i+5] == "<EOS>": res.append(stoi["<EOS>"]); i += 5
        elif s[i:i+4] == "<SOS>": res.append(stoi["<SOS>"]); i += 4
        elif s[i] in stoi: res.append(stoi[s[i]]); i += 1
        else: i += 1
    return res

def decode(l):
    return "".join([itos[i] for i in l if i in itos])

# =========================================
# 2. Model Architecture (Exact match for V2 weights)
# =========================================
class V2Block(nn.Module):
    def __init__(self, d_model, n_head=4):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_size = d_model // n_head
        
        self.mha = nn.Module()
        self.mha.in_proj = nn.Linear(d_model, 3 * d_model)
        self.mha.out_proj = nn.Linear(d_model, d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, C = x.shape
        # MHA
        res = self.norm1(x)
        qkv = self.mha.in_proj(res) # [B, T, 3*C]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        
        # Attention
        wei = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        mask = torch.tril(torch.ones(T, T, device=DEVICE))
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.mha.out_proj(out)
        
        # FF
        x = x + self.ff(self.norm2(x))
        return x

class MathTransformerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_embedding = nn.Parameter(torch.zeros(1, BLOCK_SIZE, D_MODEL))
        self.layers = nn.ModuleList([V2Block(D_MODEL) for _ in range(N_LAYER)])
        self.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embedding(idx) + self.pos_embedding[:, :T, :]
        for block in self.layers:
            x = block(x)
        logits = self.fc_out(x)
        return logits

# =========================================
# 3. Testing Suite
# =========================================
def generate(model, text, max_new_tokens=64):
    model.eval()
    idx = torch.tensor([encode(text)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if idx_next.item() == stoi.get("<EOS>", -1):
                break
    return decode(idx[0].tolist())

def verify():
    model = MathTransformerV2().to(DEVICE)
    state_dict = torch.load("/home/zzlee/models/mini_transformer_math_v2.pt", map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    print("--- Verifying Model V2 ---")
    
    # Test Addition
    add_tests = [
        (12, 34, "21+43="), # 12+34
        (123, 456, "321+654="),
        (999, 1, "999+1="),
    ]
    
    print("\n[Addition Tests]")
    for a, b, prompt in add_tests:
        gen = generate(model, prompt)
        correct = str(a+b) in gen
        print(f"Input: {a}+{b} | Prompt: {prompt} | Gen: {gen} | Correct: {correct}")

    # Test Binary
    bin_tests = [
        (13, "13="),
        (42, "42="),
        (100, "100="),
    ]
    
    print("\n[Binary Tests]")
    for n, prompt in bin_tests:
        gen = generate(model, prompt)
        correct = bin(n)[2:][::-1] in gen # The model generates binary bits reversed/stepwise
        print(f"Input: {n} | Prompt: {prompt} | Gen: {gen} | Correct: {correct}")

if __name__ == "__main__":
    verify()
