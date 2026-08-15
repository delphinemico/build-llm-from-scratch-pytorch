# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Modified and annotated by Delphine Mico, 2026, for educational study.

import torch
from chapter_04.scripts.gpt import generate_text_simple

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


# utility functions for text to token ID conversion
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<endoftext>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor # returns a tensor : batch of token IDs. Ex: tensor([[6109, 3626, 6100,  345]]) 

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens): # determines the number of iterations
        idx_cond = idx[:, -context_size:] # idx conditioned by the context
        with torch.no_grad():
            logits = model(idx_cond) # shape: (b, seq_len, vocab_size)
        logits = logits[:, -1, :] # we only care about the last token

        if top_k is not None: # i.e. top-k sampling
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition= logits < min_val,
                input= torch.tensor(float('-inf')).to(logits.device), # i.e. masking with -inf the non-topk values so that later the softmax only considers the top k values
                other= logits
            )

        if temperature > 0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim=-1) # re-normalizing the probabilities so that they sum up to 1
            idx_next = torch.multinomial(probs, num_samples=1) # i.e. multinomial sampling - model samples one token from this probability distribution
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) # i.e using greedy decoding sampling

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx   