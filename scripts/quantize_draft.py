"""ModelOpt's quantize_drafter.py plus INT4 weight-only (W4A16, group 128), the drafter format TensorRT-LLM serves on Ampere.

    python scripts/quantize_draft.py --drafter_path $LAB_DATA/heads/<head> --qformat int4_blockwise_weight_only \\
        [--quantize_lm_head] --export_path $LAB_DATA/heads/<head>_int4

"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/Model-Optimizer/examples/speculative_decoding/scripts"))
import quantize_drafter as qd  # noqa: E402

INT4 = "int4_blockwise_weight_only"

qd.SUPPORTED_QFORMATS.append(INT4)
upstream_build_quant_cfg = qd.build_quant_cfg


def build_quant_cfg(qformat: str, exclude: list[str], quantize_lm_head: bool) -> dict:
    if qformat != INT4 or not quantize_lm_head:
        return upstream_build_quant_cfg(qformat, exclude, quantize_lm_head)
    cfg = upstream_build_quant_cfg(qformat, exclude, False)
    weight = next(r for r in cfg["quant_cfg"] if r.get("quantizer_name") == "*weight_quantizer")
    cfg["quant_cfg"].append({"quantizer_name": "*lm_head*weight_quantizer", "cfg": copy.deepcopy(weight["cfg"]),
                             "enable": True})
    return cfg


qd.build_quant_cfg = build_quant_cfg
upstream_export = qd.export_quantized_state_dict


def export_quantized_state_dict(root, state_dict, dtype):
    out = upstream_export(root, state_dict, dtype)
    out.update({k: v for k, v in state_dict.items() if not v.is_floating_point()})
    return out


qd.export_quantized_state_dict = export_quantized_state_dict

if __name__ == "__main__":
    qd.main()
