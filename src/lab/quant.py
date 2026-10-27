import copy
import time

import modelopt.torch.quantization as mtq
import torch

FORMATS = {
    "int8_wo": mtq.INT8_WEIGHT_ONLY_CFG,
    "int4_rtn": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w8a8_naive": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "mxfp8": mtq.MXFP8_DEFAULT_CFG,
    "nvfp4": mtq.NVFP4_DEFAULT_CFG,
    "nvfp4_awq": mtq.NVFP4_AWQ_LITE_CFG,
}
KV_FORMATS = {"fp8": mtq.FP8_KV_CFG, "nvfp4": mtq.NVFP4_KV_CFG}

# Formats with real low-precision kernels on Ampere (sm_86), the rest need FP8/FP4 tensor cores (sm_89+)
AMPERE_DEPLOYABLE = {"int8_wo", "int4_rtn", "int4_awq", "w8a8_naive", "int8_sq"}


def make_config(base: str, exclude_layers=(), num_layers: int | None = None, kv_cache: str | None = None,
                quantize_lm_head: bool = False) -> dict:
    """Preset `base` (+ optional KV-cache format, lm_head) with whole decoder layers kept in BF16. Negative indices need `num_layers`."""
    cfg = copy.deepcopy(FORMATS[base])
    if kv_cache:
        cfg["quant_cfg"] += copy.deepcopy(KV_FORMATS[kv_cache]["quant_cfg"])
    if quantize_lm_head:
        # Presets disable "*lm_head*"; a later rule re-enables its weights in the preset's own weight format
        weight = next(r for r in cfg["quant_cfg"] if r.get("quantizer_name") == "*weight_quantizer")
        cfg["quant_cfg"].append({"quantizer_name": "*lm_head.weight_quantizer", "cfg": copy.deepcopy(weight["cfg"]),
                                 "enable": True})
    for i in exclude_layers:
        i = i % num_layers if i < 0 else i
        cfg["quant_cfg"].append({"quantizer_name": f"*layers.{i}.*", "enable": False})
    return cfg


def calib_loader(tokenizer, num_samples: int = 128, seq_len: int = 512, batch_size: int = 8, dataset: str = "cnn_dailymail",
                 apply_chat_template: bool = False):
    from modelopt.torch.utils.dataset_utils import get_dataset_dataloader

    return get_dataset_dataloader(dataset, tokenizer, batch_size=batch_size, num_samples=num_samples,
                                  max_sample_length=seq_len, device="cuda", apply_chat_template=apply_chat_template)


def calib_id(calib: dict) -> str:
    return f"{calib['dataset']}-{calib['num_samples']}x{calib['seq_len']}" + ("-chat" if calib.get("apply_chat_template") else "")


def forward_loop_from(loader):
    def forward_loop(model):
        with torch.no_grad():
            for batch in loader:
                model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

    return forward_loop


def quantize(model, config: dict, forward_loop) -> tuple[torch.nn.Module, float]:
    t0 = time.perf_counter()
    model = mtq.quantize(model, config, forward_loop)
    torch.cuda.synchronize()
    return model, time.perf_counter() - t0


def _bits(q) -> float:
    nb = q.num_bits
    return 1 + sum(nb) if isinstance(nb, tuple) else nb


def _bits_per_weight(q, in_features: int) -> float:
    bits, bs = _bits(q), q.block_sizes
    if bs and -1 in bs:
        # FP8 / E8M0 block scales are 8-bit, INT4 group scales FP16
        bits += (8 if bs.get("scale_bits") else 16) / bs[-1]
    elif q.axis is not None:
        bits += 16 / in_features
    return bits


def precision_summary(model) -> dict:
    """Storage size (quantized weights + scales, all other parameters 16-bit) and activation / KV-cache bits."""
    from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer

    total_bits, quantized, act_bits, kv_bits = 0.0, set(), 16, 16
    for name, m in model.named_modules():
        q = getattr(m, "weight_quantizer", None)
        if isinstance(q, SequentialQuantizer):
            q = q[0]  # W4A8: the INT4 stage is what's stored
        if isinstance(q, TensorQuantizer) and q.is_enabled:
            total_bits += m.weight.numel() * _bits_per_weight(q, m.weight.shape[-1])
            quantized.add(id(m.weight))
            if m.input_quantizer.is_enabled:
                act_bits = _bits(m.input_quantizer)
        if name.endswith("k_bmm_quantizer") and m.is_enabled:
            kv_bits = _bits(m)
    params = list(model.parameters())
    total_bits += sum(p.numel() * 16 for p in params if id(p) not in quantized)
    n = sum(p.numel() for p in params)
    return {"bits_per_param": round(total_bits / n, 3), "weight_gb": round(total_bits / 8e9, 2), "act_bits": act_bits,
            "kv_bits": kv_bits}
