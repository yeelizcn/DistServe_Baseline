#include <stdexcept>

#include "qwen2op.h"

#include "util/torch_utils.h"

namespace st::model {
static GptHyperParam qwenParams(int64_t vocab, int64_t positions, int64_t hidden,
    int64_t layers, int64_t q, int64_t kv, int64_t dim, int64_t ffn,
    double theta, double eps) {
    auto p = GptHyperParam::GetLlama2HyperParam(vocab, positions, hidden, layers, q, kv, dim, ffn);
    p.is_attn_qkv_biased = true;
    p.rope_theta = theta;
    p.rope_half_split = true;
    p.norm_epsilon = eps;
    p.weight_shards = 2;
    return p;
}

Qwen2Op::Qwen2Op(const int64_t vocab_size,
	               const int64_t max_position_embeddings,
	               const int64_t hidden_size,
	               const int64_t num_layers,
	               const int64_t num_q_heads,
                   const int64_t num_kv_heads,
	               const int64_t head_dim,
                   const int64_t ffn_inter_dim,
                   std::string inference_dtype,
                   const int64_t block_size,
                   const int64_t max_num_block_per_req,
                   const std::vector<int64_t> parallel_config, double rope_theta, double norm_epsilon):
    GptOpBase(inference_dtype,
        qwenParams(
            vocab_size,
            max_position_embeddings,
            hidden_size,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            ffn_inter_dim, rope_theta, norm_epsilon
        ),
        GptPagedAttnParam{
            .block_size = block_size,
            .max_num_block_per_req = max_num_block_per_req,
        },
        GptParallelismParam(parallel_config)
    ) {
};

} // namespace st::model
