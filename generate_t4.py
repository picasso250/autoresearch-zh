"""
Simple text generation from checkpoint_last.pt for the T4/Colab workflow.
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import Tokenizer


cap = torch.cuda.get_device_capability()
SUPPORTS_BF16 = cap >= (8, 0) and torch.cuda.is_bf16_supported(including_emulation=False)
LOW_PRECISION_DTYPE = torch.bfloat16 if SUPPORTS_BF16 else torch.float16
USE_FLASH_ATTN = cap >= (8, 0)

if USE_FLASH_ATTN:
    from kernels import get_kernel

    repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
    fa3 = get_kernel(repo).flash_attn_interface
else:
    fa3 = None


@dataclass
class GPTConfig:
    sequence_len: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    window_pattern: str


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        bsz, seqlen, _ = x.size()
        q = self.c_q(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.c_k(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(bsz, seqlen, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(bsz, seqlen, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        if USE_FLASH_ATTN:
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            y = y.contiguous().view(bsz, seqlen, -1)
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(config.vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)}
        )
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype=LOW_PRECISION_DTYPE), sin.to(dtype=LOW_PRECISION_DTYPE)
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx):
        _, t = idx.size()
        cos_sin = self.cos[:, :t], self.sin[:, :t]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)
        logits = self.lm_head(x).float()
        softcap = 15
        return softcap * torch.tanh(logits / softcap)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    token_ids = tokenizer.encode(prompt)
    if isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    x = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.eval()
    for _ in range(max_new_tokens):
        x_cond = x[:, -model.config.sequence_len :]
        with torch.amp.autocast(device_type="cuda", dtype=LOW_PRECISION_DTYPE):
            logits = model(x_cond)[:, -1, :]
        if temperature <= 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
    return tokenizer.decode(x[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text from checkpoint_last.pt")
    parser.add_argument("--dataset", choices=("tinystories", "tinystorieszh"), default=None)
    parser.add_argument("--checkpoint", default="checkpoint_last.pt")
    parser.add_argument("--prompt", default="从前有一只小猫")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu")
    config = GPTConfig(**payload["config"])
    dataset = args.dataset or payload.get("dataset")
    tokenizer = Tokenizer.from_directory(dataset=dataset)
    device = torch.device("cuda")
    model = GPT(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    text = generate(model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device)
    print("---")
    print(f"prompt: {args.prompt}")
    print("generation:")
    print(text)


if __name__ == "__main__":
    raise SystemExit(main())
