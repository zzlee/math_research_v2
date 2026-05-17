import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import re

# ==========================================
# 1. Configuration (Matched to trained model)
# ==========================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_MODEL = 128
N_HEAD = 4
N_LAYER = 4
BLOCK_SIZE = 64

SPECIAL_TOKENS = ["<SOS>", "<EOS>", " "]
DIGITS = "0123456789"
SYMBOLS = "+=|"
ALL_TOKENS = list(DIGITS) + list(SYMBOLS) + SPECIAL_TOKENS
vocab_size = len(ALL_TOKENS)
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
# 2. Model Architecture
# ==========================================
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(D_MODEL, head_size, bias=False)
        self.query = nn.Linear(D_MODEL, head_size, bias=False)
        self.value = nn.Linear(D_MODEL, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        tril = torch.tril(torch.ones(T, T, device=DEVICE))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, D_MODEL)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MathTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, D_MODEL)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.blocks = nn.Sequential(*[Block(D_MODEL, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

# ==========================================
# 3. Verification Logic
# ==========================================
def generate_result(model, text, max_new_tokens=64):
    model.eval()
    idx = torch.tensor([encode(text)], dtype=torch.long, device=DEVICE)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -BLOCK_SIZE:])
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
    weights_path = "/home/zzlee/math_research_v2/math_model.pth"
    model = MathTransformer().to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    print("Weights loaded successfully.")
    
    add_ex = verify_addition(model)
    bin_ex = verify_binary(model)
    
    print("\n--- Examples ---")
    for ex in add_ex + bin_ex:
        status = "✅" if ex["correct"] else "❌"
        print(f"{status} {ex['type']} | {ex['input']} | Pred: {ex['pred']} | Actual: {ex['actual']} | Gen: {ex['gen']}")
