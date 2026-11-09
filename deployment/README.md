# 双节点 Qwen2.5 适配（待 GPU 验收）

目标：node2 两张 A6000 做 Prefill，node1 两张 A5000 做 Decode；每端 TP=2、PP=1、FP16、FCFS。模型为 Qwen/Qwen2.5-7B-Instruct，来自用户指定项目 README 第 183 行。本目录不是已完成服务器验收的声明。

## 实现边界

保留 DistServe 调度、请求对象、分页 KV 和 SwiftTransformer 执行引擎。新增显式 cross_node 配置；未启用时继续原 IPC 路径。Decode 调度器按需启动两端 send/recv；全部 rank 完成后才释放 Prefill 缓存。CUDA 张量直接交给 PyTorch NCCL，不经过 Python CPU 序列化。是否走 RDMA/GPUDirect 取决于服务器网络与 NCCL 配置，不能从使用 NCCL 推断。

当前逐块、逐 K/V 传输，保守同步以保证完成边界；这是适配实现，性能不等同论文传输实现，尚未优化通信重叠。传输失败抛错并终止本次 smoke，不能继续收集性能结果。原工程取消请求尚未实现，不宣称故障恢复或取消验收通过。

Qwen 分支支持原生稠密注意力、未缩放 RoPE、TP=1/2、PP=1。新增 QKV bias、half-split RoPE、配置化 theta/epsilon、两份权重分片，以及 7:1 GQA 内核分组。拒绝滑动窗口、RoPE scaling 或其他尺寸架构。保持原始 HF Q/K 排列，不把 Qwen 当作 LLaMA 名称别名。

## 服务器准备顺序

1. 两端执行 `bash deployment/inspect_node.sh`，确认 GPU 分配、驱动、网络、磁盘和依赖；目前没有这些检查结果。
2. 在独立 Linux 环境复用根目录 environment.yml（Python 3.10、PyTorch 2.2.2、Ray 2.10），安装 `python -m pip install -r deployment/requirements-qwen.txt`。原环境文件版本较旧，需根据驱动评估，不升级系统驱动。
3. 执行 `bash deployment/bootstrap_backend.sh`：取得固定 xformers 子模块、编译修改后的 SwiftTransformer（sm_86），安装 DistServe。必须上传整个 SwiftTransformer，不能用未修改的 upstream 覆盖它。构建尚未在本机验证。
4. 确认一个模型的 40 位 commit；在两端下载相同 revision，不下载数据集：
   `python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', revision='<40位commit>'))"`
   使用返回的 snapshots/<commit> 目录。当前启动入口要求两端绝对模型路径相同；若 HOME 不同，可指定相同 cache_dir。Controller 也需能读 tokenizer/config。不能覆盖该不可变快照的权重文件。
5. 两端仅暴露指定的两张卡，配置 GPU_IDS、NODE_IP、ROLE 后执行 start_ray.sh。Prefill 端额外设置 RAY_PORT，Decode 端设置 RAY_HEAD。不要在已有共享 Ray 服务上执行 ray stop；应使用项目专用 Ray 集群。脚本拒绝已被计算进程占用的 GPU。
6. 在 Prefill 节点连接 Ray 后，用 `ray.nodes()` 查询实际 NodeID。复制 two-node.example.json 为本地配置，填写两个 NodeID、ray_address、模型 revision/path，以及 `tcp://<Prefill可达IP>:<独占空闲端口>`。NCCL 网卡变量只根据实测填写，须在两端 Ray 启动前设置；代码不猜测网卡。
7. 从仓库根目录运行 `python -m deployment.run_smoke deployment/two-node.local.json`。入口套用模型 chat template，检查输入输出长度，执行一个短请求，有超时和非空输出校验，结束后销毁本次 Worker。输出 token_ids 用于数值对照，不等同正确性验收。

## 必须继续完成的验收

- 编译修改后的 CUDA 库；验证两端配置和权重 SHA256 相同。
- NCCL 最小传输及实际网卡确认；检查接收 KV 与发送 KV 的 shape、dtype、数值完全一致。
- 与同 revision 的 Hugging Face greedy 推理对照首 token 和连续输出（允许接近相同 logits 的数值分歧需逐项分析）；分别覆盖短/长输入、TP=1/2。
- 单请求、顺序、并发、故障清理、显存边界验收之后，才开始 WildChat 正式测量。
- 本次未新增数据集下载/预处理或正式负载驱动；原计划中的 WildChat 本地处理与指标采集仍待完成。

## 本地检查

`python tests/test_two_node_cpu.py` 检查配置约束和实际 QKV 权重两分片的内容与偏置顺序；需要 CPU PyTorch/tqdm。Python 语法检查不等同 CUDA 编译或 NCCL 测试。

## 停止和回滚

smoke 完成或异常会销毁自己创建的 Worker。Ray 守护进程可保留；只有确认本项目独占该 Ray 集群后才手动停止。回滚时恢复原 DistServe 文件并重新编译原 SwiftTransformer；不要将旧二进制与新增 Qwen Python 入口混用。模型与数据文件未被修改。
