import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert(d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # trainable weight matrix of dimensions (d_in, d_out)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias) # trainable weight matrix of dimensions (d_in, d_out)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias) # trainable weight matrix of dimensions (d_in, d_out)
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape # Note that num_tokens <= context_length (context_length is the maximum sequence length, whereas num_tokens is the actual seq length in the batch)
        queries = self.W_query(x) # this is equivalent to: Q = X . Wq (i.e. Projection Q and has dim (b, n, d_out) )
        keys = self.W_key(x) # this is equivalent to: K = X . Wk (i.e. Projection K and has dim (b, n, d_out) )
        values = self.W_value(x) # this is equivalent to: V = X . Wv (i.e. Projection V and has dim (b, n, d_out) )
        

        # Here, the last dim 'd_out' is now split into self.num_heads and self.head_dim (d_out = num_heads * head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) # queries dim: (b, n, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # keys dim: (b, n, num_heads, head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) # values dim: (b, n, num_heads, head_dim)

        # Here, we swap n, num_heads dim positions so that we can keep (b, num_heads) fixed, while we operate matrix multiplications using the last 2 dims (n, head_dim)
        queries = queries.transpose(1,2) # queries dim: (b, num_heads, n, head_dim)
        keys = keys.transpose(1,2) # keys dim: (b, num_heads, n, head_dim)
        values = values.transpose(1,2) # values dim: (b, num_heads, n, head_dim)
        d_k = keys.shape[-1] # i.e. head_dim

        attn_scores = queries @ keys.transpose(2,3) # attn_scores is a (batch_size, num_heads) 'number' of square matrices --> dim: (b,num_heads, n,n)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) # applying the causal mask
        attn_weights = torch.softmax(attn_scores/d_k**0.5, dim=-1) # attn_weights dim: (b,num_heads, n,n)
        attn_weights = self.dropout(attn_weights) # applying the dropout mask --> # attn_weights dim: (b,num_heads, n,n)
        context_vec = (attn_weights @ values).transpose(1,2) # (attn_weights @ values) dim: (b,num_heads, n, head_dim) --> context_vec dim: (b,n, num_head, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # UNSPLITTING: num_heads and head_dim are now combined BACK into d_out. 'contiguous()' ensures a compatible memory layout after transpose
        context_vec = self.out_proj(context_vec) # applies a learned linear transformation to the concatenated head outputs, allowing features from different heads to be mixed
        return context_vec # dim:(b, n, d_out)