import torch

from distserve.config import ModelConfig, ParallelConfig, CacheConfig


def get_model_op(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    cache_config: CacheConfig,
):
    model_name = model_config.model
    model_type = model_config.hf_config.model_type
    match model_type:
      case 'qwen2':
        cfg = model_config.hf_config
        if cfg.rope_scaling or getattr(cfg, 'use_sliding_window', False):
            raise ValueError("Qwen adapter supports dense attention and unscaled RoPE only")
        if (cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.num_hidden_layers, cfg.intermediate_size) != (3584, 28, 4, 28, 18944):
            raise ValueError("Adapter is restricted to Qwen2.5-7B architecture")
        if parallel_config.tensor_parallel_size not in (1, 2) or parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Qwen adapter supports TP=1/2, PP=1")
        return torch.classes.gpt_ops.Qwen2Op(
            cfg.vocab_size, model_config.get_max_model_len(), cfg.hidden_size,
            cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
            model_config.get_head_size(), cfg.intermediate_size, model_config.dtype,
            cache_config.block_size, cache_config.max_num_blocks_per_req,
            parallel_config.to_list(), float(cfg.rope_theta), float(cfg.rms_norm_eps))
      case 'opt':
        return torch.classes.gpt_ops.OptOp(
            model_config.hf_config.vocab_size,
            model_config.get_max_model_len(),
            model_config.get_hidden_size(),
            model_config.get_num_layers(),
            model_config.get_num_heads(),
            model_config.get_head_size(),
            model_config.dtype,
            cache_config.block_size,
            cache_config.max_num_blocks_per_req,
            parallel_config.to_list(),
        )
      case 'gpt2':
        return torch.classes.gpt_ops.Gpt2Op(
            model_config.hf_config.vocab_size,
            model_config.get_max_model_len(),
            model_config.get_hidden_size(),
            model_config.get_num_layers(),
            model_config.get_num_heads(),
            model_config.get_head_size(),
            model_config.dtype,
            cache_config.block_size,
            cache_config.max_num_blocks_per_req,
            parallel_config.to_list(),
        )
      case 'llama':
        return torch.classes.gpt_ops.Llama2Op(
            model_config.hf_config.vocab_size,
            model_config.get_max_model_len(),
            model_config.get_hidden_size(),
            model_config.get_num_layers(),
            model_config.get_q_heads(),
            model_config.get_num_heads(),
            model_config.get_head_size(),
            model_config.get_ffn_inter_dim(),
            model_config.dtype,
            cache_config.block_size,
            cache_config.max_num_blocks_per_req,
            parallel_config.to_list(),
        )
    raise NotImplementedError(f"model {model_name} not supported")
