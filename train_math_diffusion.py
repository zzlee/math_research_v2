import torch
import torch.nn as nn
from torch.nn import functional as F
import random

# ==========================================
# 1. 配置與超參數
# ==========================================
BATCH_SIZE = 64
BLOCK_SIZE = 64
MAX_ITERS = 50000
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
D_MODEL = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

# Token 字典
# 使用列表定義特殊 Token，確保它們被視為單一單位
SPECIAL_TOKENS = ["<EOS>", "<SOS>", " ", "<MASK>"]
DIGITS = "0123456789"
SYMBOLS = "+=|"
ALL_TOKENS = list(DIGITS) + list(SYMBOLS) + SPECIAL_TOKENS
vocab_size = len(ALL_TOKENS)
stoi = { token: i for i, token in enumerate(ALL_TOKENS) }
itos = { i: token for i, token in enumerate(ALL_TOKENS) }

def encode(s):
    # 簡單的 Tokenizer: 優先處理特殊 Token，然後處理單個字元
    # 在此實驗中，我們假設輸入是標準格式，直接拆分字元即可
    # 但需要處理 <EOS> 等特殊標記
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
# 2. 數據生成 (CoT 策略)
# ==========================================
def generate_addition_cot():
    a = random.randint(1, 999)
    b = random.randint(1, 999)
    s_a = str(a)[::-1]
    s_b = str(b)[::-1]
    res = a + b
    res_str = str(res)
    cot = ""
    carry = 0
    for i in range(max(len(s_a), len(s_b))):
        d1 = int(s_a[i]) if i < len(s_a) else 0
        d2 = int(s_b[i]) if i < len(s_b) else 0
        sum_val = d1 + d2 + carry
        carry = sum_val // 10
        cot += f"{d1}+{d2}={sum_val%10}|"
    full_seq = f"{s_a}+{s_b}= {cot}result={res_str}"
    return full_seq

def generate_binary_cot():
    n = random.randint(1, 100)
    temp_n = n
    cot = ""
    res_bits = []
    while temp_n > 0:
        q = temp_n // 2
        r = temp_n % 2
        cot += f"{temp_n}/2={q} r{r}|"
        res_bits.append(str(r))
        temp_n = q
    res_str = "".join(res_bits)
    full_seq = f"{n}= {cot}result={res_str}"
    return full_seq

def get_batch():
    data = []
    for _ in range(BATCH_SIZE):
        data.append(generate_addition_cot())

    encoded = [encode(s) + [stoi['<EOS>']] for s in data]
    padded = [seq + [0]*(BLOCK_SIZE - len(seq)) if len(seq) < BLOCK_SIZE else seq[:BLOCK_SIZE] for seq in encoded]
    x = torch.tensor(padded, dtype=torch.long, device=DEVICE)
    y = x.clone()

    mask_token_id = stoi["<MASK>"]
    for i in range(BATCH_SIZE):
        seq = encoded[i]
        seq_len = min(len(seq), BLOCK_SIZE)
        # Find prompt length (everything up to and including '= ' or just '=')
        prompt_len = 0
        for j in range(seq_len):
            if padded[i][j] == stoi['=']:
                prompt_len = j + 2 if j + 1 < seq_len and padded[i][j+1] == stoi[' '] else j + 1
                break
        if prompt_len == 0 or prompt_len >= seq_len:
            prompt_len = seq_len // 2 # fallback

        # Randomly mask a portion of the non-prompt tokens
        num_to_mask = seq_len - prompt_len
        if num_to_mask > 0:
            mask_ratio = random.uniform(0.1, 1.0)
            mask_count = max(1, int(num_to_mask * mask_ratio))

            mask_indices = random.sample(range(prompt_len, seq_len), mask_count)
            x[i, mask_indices] = mask_token_id

            # For unmasked positions in y, set to ignore index
            y[i, :prompt_len] = -100
            for j in range(prompt_len, seq_len):
                if j not in mask_indices:
                    y[i, j] = -100

        # padding tokens should also be ignored in loss
        y[i, seq_len:] = -100

    return x, y

# ==========================================
# 3. 模型定義 (Tiny Transformer)
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

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)

        return logits, loss

# ==========================================
# 4. 訓練迴圈
# ==========================================
model = MathTransformer().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training on {DEVICE}...")
for i in range(MAX_ITERS):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(f"iter {i}: loss {loss.item():.4f}")

# Save the model
torch.save(model.state_dict(), "math_model.pth")
print("\nModel saved to math_model.pth")

# ==========================================
# 5. 測試與驗證
# ==========================================
def generate_result(text, max_new_tokens=64):
    model.eval()
    # Initialize the sequence with the prompt and fill the rest with <MASK>
    prompt_encoded = encode(text)
    seq_len = min(len(prompt_encoded) + max_new_tokens, BLOCK_SIZE)
    idx = prompt_encoded + [stoi["<MASK>"]] * (seq_len - len(prompt_encoded))
    idx = torch.tensor([idx], dtype=torch.long, device=DEVICE)

    # Text Diffusion: Iterative decoding
    T = 10 # Number of diffusion steps
    mask_token_id = stoi["<MASK>"]
    prompt_len = len(prompt_encoded)

    for step in range(T):
        logits, _ = model(idx)
        # Get probabilities for all tokens
        probs = F.softmax(logits, dim=-1) # [1, seq_len, vocab_size]

        # Get the highest probability prediction for each position
        max_probs, preds = torch.max(probs, dim=-1) # [1, seq_len]

        # Calculate how many tokens to unmask at this step
        # Linear schedule: unmask a proportion of the unknown tokens
        ratio = (step + 1) / T

        # We only care about positions that are currently masked
        is_masked = (idx == mask_token_id)
        if not is_masked.any():
            break

        # Get confidence of predictions at masked positions
        confidence = max_probs.clone()
        confidence[~is_masked] = -1.0 # Ignore unmasked positions

        # Determine number of tokens to unmask this step
        total_masked = is_masked.sum().item()
        num_to_unmask = max(1, int(total_masked * (1.0 / (T - step))))
        if step == T - 1:
            num_to_unmask = total_masked # Unmask all remaining at the last step

        # Get indices of top confidence predictions
        _, top_indices = torch.topk(confidence.view(-1), num_to_unmask)

        # Update the sequence
        for i in top_indices:
            idx[0, i] = preds[0, i]

        # Stop early if generated <EOS>
        if stoi["<EOS>"] in idx[0, prompt_len:]:
            # Find the first EOS and truncate
            eos_pos = (idx[0, prompt_len:] == stoi["<EOS>"]).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                idx = idx[:, :prompt_len + eos_pos[0].item() + 1]
                break

    return decode(idx[0].tolist())

print("\n--- Testing Math-CoT Transformer ---")
test_cases = [
    "21+43=", # 12+34
    "31+21=", # 13+21
    "13="      # binary of 13
]
for tc in test_cases:
    print(f"Input: {tc} | Generated: {generate_result(tc)}")
