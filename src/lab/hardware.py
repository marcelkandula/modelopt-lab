# RTX A6000 datasheet
A6000_MEM_BW_GBS = 768.0


def kv_cache_bytes_per_token(config, bytes_per_elem: int = 2) -> int:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * bytes_per_elem


def decode_ceiling_tok_s(bytes_read_per_token: float, bw_gbs: float = A6000_MEM_BW_GBS) -> float:
    return bw_gbs * 1e9 / bytes_read_per_token
