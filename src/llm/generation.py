import torch
import time

def generate(model, input_ids: torch.Tensor, past_key_values, max_new_tokens: int = 50):
    device = model.model.embed_tokens.weight.device
    origin_len = input_ids.shape[-1]
    input_ids = input_ids.to(device)
    output_ids = input_ids.clone()
    next_token = input_ids

    prefill_time = 0
    decode_time = 0

    with torch.no_grad():
        for i in range(max_new_tokens):
            t0 = time.time()
            out = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            t1 = time.time()

            if i == 0:
                prefill_time = (t1 - t0) * 1000
            else:
                decode_time += (t1 - t0) * 1000

            logits = out.logits[:, -1, :]
            token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, token], dim=-1)
            next_token = token.to(device)

            eos_ids = model.config.eos_token_id
            if eos_ids is not None:
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
                if token.item() in eos_ids:
                    break

    generated_ids = output_ids[:, origin_len:]
    return generated_ids, prefill_time, decode_time