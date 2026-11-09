# DistServe

## 本仓库：双节点 Qwen2.5 Baseline 适配

基于 LLMServe/DistServe，目标是两张 A6000 执行 Prefill、两张 A5000 执行 Decode，使用 Qwen/Qwen2.5-7B-Instruct。新增固定双节点放置、NCCL KV Cache 传输和 SwiftTransformer Qwen 执行支持。

**验证状态：CPU 测试通过；尚未完成 Linux/CUDA 编译、真实 NCCL 通信和模型数值验收。不能视为已经跑通的性能复现结果。**

请先阅读 [双节点部署说明](deployment/README.md)。修改后的后端源码位于 `SwiftTransformer/`，不要用下面原版说明中的再次克隆操作覆盖它。实验记录位于 `复现指导/复现实验日志.docx`。

以下保留上游项目介绍与原版使用说明。

DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding
computation. Existing LLM serving systems colocate the two
phases and batch the computation of prefill and decoding
across all users and requests. We find that this strategy not
only leads to strong prefill-decoding interferences but also
couples the resource allocation and parallelism plans for both
phases. In DistServe, you can simply set the parallelism configs and scheduling strategies for the two phases and it will work just like a single instance which handles the KV-Cache communication and memory management automatically. 

It utilizes a high-performance C++ Transformer inference library [SwiftTransformer](https://github.com/LLMServe/SwiftTransformer) as the execution backend, which supports many features like model/pipeline parallelism, FlashAttention, Continuous Batching, and PagedAttention.

It supports:
- GPT-2 (gpt2, gpt2-xl, ...)
- OPT (facebook/opt-1.3b, facebook/opt-6.7b, ...)
- LLaMA2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b, ...)

## Build && Install
```shell
# clone the project
git clone https://github.com/LLMServe/DistServe.git && cd DistServe

# setup the distserve conda environment
conda env create -f environment.yml && conda activate distserve

# clone and build the SwiftTransformer library  
git clone https://github.com/LLMServe/SwiftTransformer.git && cd SwiftTransformer && git submodule update --init --recursive
cmake -B build && cmake --build build -j$(nproc)
cd ..

# install distserve
pip install -e .
```

## Launching

### Launch Ray Cluster

DistServe relies on [Ray](https://ray.io) to implement distributed workers. If you do not launch a Ray runtime in advance, it will automatically initiate a cluster consisting of all the gpus on the current node. You may need to start the Ray runtime manually in advance if you want to use multiple nodes for inference.

### Run offline example

DistServe requires at least two GPUs to play with. We provide an offline inference example in `examples/offline.py`.

### Run online example

To run online inference, you need to launch the DistServe API server, see the comments in `distserve/api_server/distserve_api_server.py`.

Then launch the client example in `examples/online.py`.

### Evaluation

To reproduce all the experiments in our paper, please follow the [guidance](./evaluation/README.md).

## Citation
If you use DistServe for your research, please cite our [paper](https://arxiv.org/abs/2401.09670):
```
@misc{zhong2024distserve,
      title={DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving}, 
      author={Yinmin Zhong and Shengyu Liu and Junda Chen and Jianbo Hu and Yibo Zhu and Xuanzhe Liu and Xin Jin and Hao Zhang},
      year={2024},
      eprint={2401.09670},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
