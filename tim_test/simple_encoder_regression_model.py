# Here’s a minimal, practical PyTorch setup that takes encoder hidden states → pooled summary → regression output. I show three pooling options: CLS, mean, and attention pooling (mask-aware).
# Source: https://chatgpt.com/g/g-p-67acf4d73e10819199af60a18bf58fb9-ai-ml/c/6914c5b7-f14c-8332-8240-f7f0a29fb21c

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------- Pooling modules -------

class MeanPool(nn.Module):
    def forward(self, h, mask):
        # h: [B, T, D], mask: [B, T] with 1 for real tokens, 0 for padding
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        masked_h = h * mask.unsqueeze(-1)                     # [B, T, D]
        return masked_h.sum(dim=1) / lengths                  # [B, D]

class CLSPool(nn.Module):
    def forward(self, h, mask):
        # Assumes first token is a [CLS]-like token
        return h[:, 0, :]  # [B, D]

class AttnPool(nn.Module):
    """
    Learn weights α_i = softmax(v^T tanh(W h_i)) (mask-aware),
    then z = Σ α_i h_i
    """
    def __init__(self, d_model, d_attn=128):
        super().__init__()
        self.W = nn.Linear(d_model, d_attn)
        self.v = nn.Linear(d_attn, 1, bias=False)

    def forward(self, h, mask):
        # h: [B, T, D], mask: [B, T]
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)    # [B, T]
        scores = scores.masked_fill(mask == 0, -1e9)
        alpha = scores.softmax(dim=-1)                        # [B, T]
        z = (alpha.unsqueeze(-1) * h).sum(dim=1)              # [B, D]
        return z

# ------- Example encoder-agnostic regressor head -------

class SequenceRegressor(nn.Module):
    def __init__(self, d_model, pooling="mean"):
        super().__init__()
        if pooling == "mean":
            self.pool = MeanPool()
        elif pooling == "cls":
            self.pool = CLSPool()
        elif pooling == "attn":
            self.pool = AttnPool(d_model)
        else:
            raise ValueError("pooling must be one of: 'mean' | 'cls' | 'attn'")

        # A small MLP regressor; adjust width/depth as needed
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model, 1)  # scalar regression
        )

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [B, T, D] from your encoder (contextualized per-token states)
        attention_mask: [B, T] with 1 for real tokens, 0 for padding
        """
        z = self.pool(hidden_states, attention_mask)  # [B, D]
        yhat = self.mlp(z).squeeze(-1)               # [B]
        return yhat

# ------- How to plug in an encoder -------

# Option A: HuggingFace encoder (e.g., BERT/Longformer/etc.)
# from transformers import AutoModel, AutoTokenizer
# tok = AutoTokenizer.from_pretrained("bert-base-uncased")
# enc = AutoModel.from_pretrained("bert-base-uncased")  # returns last_hidden_state: [B, T, D]
# d_model = enc.config.hidden_size

# Option B: Any encoder you already have that returns [B, T, D] + takes attention_mask

# ------- End-to-end forward + loss example -------

# Dummy shapes (replace with your real batch)
B, T, D = 8, 128, 768
hidden_states = torch.randn(B, T, D)       # output from your encoder
attention_mask = (torch.rand(B, T) > 0.1).long()  # 1=token, 0=pad
targets = torch.randn(B)                   # your regression targets

model = SequenceRegressor(d_model=D, pooling="attn")  # "mean" | "cls" | "attn"
pred = model(hidden_states, attention_mask)           # [B]

loss = F.mse_loss(pred, targets)
loss.backward()
# optimizer.step(), etc.
