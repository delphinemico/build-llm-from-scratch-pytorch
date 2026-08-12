import torch
import torch.nn as nn
from chapter_03.scripts.attention import MultiHeadAttention

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # Here we use the 'biased' Population Variance (i.e. /n instead of /n-1)
        norm_x =  (x-mean)/torch.sqrt(var+self.eps) # We add a small positive eps to avoid dividing by zero 
        return self.scale * norm_x + self.shift # Where scale and shift are two trainable parameters that the LLM automatically adjusts during training  

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), # Explands to a higher dim (4 x emb_dim)
            nn.GELU(), # applies a non-liear transformation (GELU) - Not that we could also use our own custom GELU() class 
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]) # contracts back to initial dim: emb_dim
        )

    def forward(self, x):
        return self.layers(x)

import torch.nn as nn
from chapter_03.scripts.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # in the sum, x corresponds to the output, and shortcut corresponds to the initial value of x

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return  x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]) # tok_embedding_layer
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"]) # pos_embedding_layer
        self.drop_emb = nn.Dropout(cfg["drop_rate"]) # Dropout layer with dropout rate p
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]) # *: unpacks the Transformer blocks and stacks them a number n_layers of times
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False) # This is the LM Head 

    def forward(self, in_idx): # in_idx: Batch of Input Token IDs with shape (b, num_tokens=seq_len)
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
            )
        x = tok_embeds + pos_embeds # this is the Input Embeddings with shape (b, seq_len, emb_dim)
        x = self.drop_emb(x) # shape: (b, seq_len, emb_dim)
        x = self.trf_blocks(x) # shape: (b, seq_len, emb_dim)
        x = self.final_norm(x) # shape: (b, seq_len, emb_dim)
        logits = self.out_head(x) # raw, unnormalized logits - shape: (b, seq_len, vocab_size)
        return logits

def generate_text_simple(model, idx, max_new_tokens, context_size): # idx shape: (b, curr_seq_len) = (b, initial_seq_len)
    for _ in range(max_new_tokens): # repeats a number 'maxnew_tokens' of iterations
        idx_cond =  idx[:, -context_size:] # idx conditioned context - shape: (b, seq_len)
        with torch.no_grad():
            logits = model(idx_cond) # shape: (b, seq_len, vocab_size)
        logits = logits[:, -1, :] # shape: (b, 1, vocab_size) ~ (b, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # This uses 'greedy decoding' (i.e. choose the highest-probability next token)
        idx = torch.cat((idx, idx_next), dim=1) # appends next-token to the sequence
    return idx # returns a (batch of) series of token IDs - shape: (b, final_seq_len) where final_seq_len ~ initial_seq_len + max_new_tokens

GPT2_cfg_small = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False    
}

GPT2_cfg_medium = GPT2_cfg_small.copy()
GPT2_cfg_medium.update({"emb_dim": 1024, "n_heads": 16, "n_layers":24})

GPT2_cfg_large = GPT2_cfg_small.copy()
GPT2_cfg_large.update({"emb_dim": 1280, "n_heads": 20, "n_layers":36})

GPT2_cfg_XL = GPT2_cfg_small.copy()
GPT2_cfg_XL.update({"emb_dim": 1600, "n_heads": 25, "n_layers":48})
