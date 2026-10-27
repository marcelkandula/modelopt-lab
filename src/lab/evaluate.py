import functools
import math

import torch
import torch.nn.functional as F


@functools.cache
def _wikitext(tokenizer_id: str):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = AutoTokenizer.from_pretrained(tokenizer_id)(text, return_tensors="pt").input_ids
    return text, ids


@torch.no_grad()
def wikitext_ppl(model, tokenizer_id: str, max_len: int = 2048, stride: int = 1024) -> dict:
    """Sliding-window WikiText-2 test perplexity."""
    text, ids = _wikitext(tokenizer_id)
    device = next(model.parameters()).device
    total_nll, n_scored, prev_end = 0.0, 0, 0
    for begin in range(0, ids.shape[1], stride):
        end = min(begin + max_len, ids.shape[1])
        window = ids[:, begin:end].to(device)
        nll = F.cross_entropy(model(window).logits[0, :-1].float(), window[0, 1:], reduction="none")
        new = end - max(prev_end, begin + 1)
        total_nll += nll[-new:].sum().item()
        n_scored += new
        prev_end = end
        if end == ids.shape[1]:
            break
    return {
        "token_ppl": math.exp(total_nll / n_scored),
        "word_ppl": math.exp(total_nll / len(text.split())),
        "bits_per_byte": total_nll / math.log(2) / len(text.encode()),
    }


def lm_eval_task(model, tokenizer, task: str, num_fewshot: int | None = None, limit: int | None = None,
                 batch_size: int = 8, max_length: int = 4096) -> dict:
    """One lm-eval task (or group, e.g. mmlu) on an in-memory model -> {"acc,none": ..., "acc_stderr,none": ...}."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=max_length)
    out = lm_eval.simple_evaluate(model=lm, tasks=[task], num_fewshot=num_fewshot, limit=limit, log_samples=False)
    return {k: v for k, v in out["results"][task].items() if isinstance(v, (int, float))}


def api_eval_task(base_url: str, served_name: str, tokenizer: str, task: str, num_fewshot: int | None = None,
                  num_concurrent: int = 32, limit: int | None = None) -> dict:
    """The same lm-eval task against an OpenAI-compatible /v1/completions server (real kernels)."""
    import lm_eval

    args = {"base_url": f"{base_url}/v1/completions", "model": served_name, "tokenizer": tokenizer,
            "tokenizer_backend": "huggingface", "tokenized_requests": False, "num_concurrent": num_concurrent,
            "max_retries": 3}
    out = lm_eval.simple_evaluate(model="local-completions", model_args=args, tasks=[task], num_fewshot=num_fewshot,
                                  limit=limit, log_samples=False)
    return {k: v for k, v in out["results"][task].items() if isinstance(v, (int, float))}
