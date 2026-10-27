"""Let TensorRT-LLM 1.0 (PyTorch backend) run a quantized lm_head.

    $LAB_DATA/venvs/trtllm/bin/python scripts/trtllm_lm_head_patch.py <trtllm-serve args>   # patched trtllm-serve

"""

import sys

import torch
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm._torch.modules.linear import W4A16_AWQ_LinearMethod


def apply():
    original = DecoderModelForCausalLM.__post_init__

    def __post_init__(self):
        qc = self.model_config.quant_config
        if qc is not None and qc.quant_algo is not None and not qc.is_module_excluded_from_quantization("lm_head"):
            # Drop the BF16 weight LMHead created in __init__; the original __post_init__ then creates quantized weights
            del self.lm_head.weight
            self.lm_head.quant_config = qc
            self.lm_head._weights_created = False
        original(self)

    DecoderModelForCausalLM.__post_init__ = __post_init__

    # The INT4-AWQ loader requires a pre_quant_scale for every unfused Linear, the export may have folded lm_head's into
    # the final norm instead, so a missing one means "multiply by 1"
    load = W4A16_AWQ_LinearMethod.load_weights_vanilla

    def load_weights_vanilla(self, module, weights):
        if "pre_quant_scale" not in weights[0]:
            weights = [{**weights[0], "pre_quant_scale": torch.ones(module.in_features, dtype=module.dtype)}]
        load(self, module, weights)

    W4A16_AWQ_LinearMethod.load_weights_vanilla = load_weights_vanilla


if __name__ == "__main__":
    apply()
    from tensorrt_llm.commands.serve import main

    sys.argv[0] = "trtllm-serve"
    sys.exit(main())
