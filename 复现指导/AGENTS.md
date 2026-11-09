# AGENTS.md --- DistServe 双节点 PD 分离 Baseline 部署与复现规范

> 本文件是 Coding Agent
> 在修改代码、连接服务器和启动实验前必须阅读并遵守的项目级指导。
> 目标是在两台异构 GPU 服务器上部署一个基于 DistServe 的
> Prefill-Decode（PD）分离 Baseline。
> 本文件描述的是**部署和复现要求**，不是对当前仓库实现情况的预判。所有实现细节必须以仓库代码、实际硬件检查结果和用户确认的信息为准。

------------------------------------------------------------------------

## 0. 项目目标与边界

### 0.1 总体目标

基于 DistServe
原始实现，完成适配当前硬件集群的必要修改，并在两个物理服务器节点上运行一个端到端的
PD 分离推理服务，作为后续研究工作的 Baseline。

固定部署目标：

  --------------------------------------------------------------------------------
  逻辑节点      GPU 集群          分配 GPU 数量 运行角色      Instance
  ------------- ------------- ----------------- ------------- --------------------
  `node1`       NVIDIA A5000         2 张 A5000 Decode        `Decode Instance`
                集群                                          

  `node2`       NVIDIA A6000         2 张 A6000 Prefill       `Prefill Instance`
                集群                                          
  --------------------------------------------------------------------------------

其他约束：

-   模型：Qwen-7B。具体 Hugging Face 仓库 ID、模型变体和 revision
    必须在下载前确认，不得擅自替换成 Qwen1.5、Qwen2、Chat 或 Instruct
    版本。
-   数据集：[`allenai/WildChat-1M`](https://huggingface.co/datasets/allenai/WildChat-1M)。
-   模型权重：在服务器端下载并加载。Prefill 和 Decode 两个 Instance
    都需要可访问一份完整模型权重；可以分别下载，也可以在确认共享存储或安全传输方式后复用。
-   数据集：在用户本地下载、检查、预处理和生成泊松到达时间，再将处理后的
    workload 文件上传到服务器。**不要在服务器上重新下载 WildChat-1M。**
-   SSH 地址、用户名、端口、密钥和其他连接方式由用户在 Agent
    实际执行远程工作时提供。当前不得猜测或填写这些信息。

### 0.2 复现范围

本项目目标是"**在指定异构硬件和指定 workload 上运行 DistServe-based
Baseline**"，不应宣称直接复现了论文中使用 OPT 模型、A100 集群和 ShareGPT
等设置得到的原始性能数字。

必须区分：

1.  **DistServe 原有机制**：Prefill/Decode
    分离、两个阶段独立管理模型副本、请求编排、KV Cache
    跨阶段传输、各阶段的 GPU Worker 执行等。
2.  **本项目固定的实验约束**：只有一个 Prefill Instance 和一个 Decode
    Instance；Prefill 固定使用 `node2` 的两张 A6000；Decode 固定使用
    `node1` 的两张 A5000；模型为 Qwen-7B；workload 来自 WildChat-1M。
3.  **为适配环境所做的修改**：例如节点发现、GPU 绑定、Ray
    初始化、网络接口选择、模型适配和固定 Placement
    配置。每项修改都必须记录原因、影响和验证结果。

不得为了"跑通"而静默改变 DistServe 的调度语义、KV Cache
传输方式或请求执行流程。

------------------------------------------------------------------------

## 1. 统一术语与目标拓扑

### 1.1 术语

在代码、配置、日志和文档中统一使用以下术语：

-   **Node**：一台物理服务器。`node1` 和 `node2`
    是本项目的逻辑主机名，不代表 IP 地址。
-   **Instance**：管理一份完整模型权重副本的逻辑推理资源单元。一个
    Instance 可以由多张 GPU 共同提供计算资源。
-   **Worker**：Instance 内部实际执行模型计算、管理对应 GPU
    资源的工作进程。DistServe 原文实现使用 Ray Actor 承载 GPU
    Worker；当前仓库是否仍采用相同实现，必须以代码检查为准。
-   **Prefill Instance**：只负责 Prompt Prefill 阶段。
-   **Decode Instance**：只负责自回归 Decode 阶段。
-   **KV Cache Transfer**：将 Prefill 阶段生成的 KV Cache
    及必要请求状态传递给 Decode 阶段。

**不要把 Node、Instance 和 Worker 混为一谈。** 本项目的预期关系是：两个
Node、两个 Instance；每个 Instance 使用两张 GPU，并由其内部 Worker
执行计算。Worker 数量、进程数量和 GPU
映射必须从实际代码和运行配置中确定，不能直接假设"一张 GPU 必然对应一个
Worker"。

### 1.2 固定拓扑

``` text
                         Client / Workload Driver
                                   |
                                   v
                         DistServe Controller
                                   |
                       Dispatch to Prefill Instance
                                   |
              +--------------------+--------------------+
              |                                         |
              v                                         |
       node2: A6000 cluster                              |
       2 selected A6000 GPUs                             |
       +--------------------------------+                |
       |         Prefill Instance       |                |
       |                                |                |
       |   Prefill Worker(s)            |                |
       |   - Qwen-7B model copy         |                |
       |   - Prompt Prefill             |                |
       |   - KV Cache generation        |                |
       +--------------------------------+                |
              |                                         |
              | KV Cache + request state                |
              | cross-node transfer                     |
              v                                         |
       node1: A5000 cluster                              |
       2 selected A5000 GPUs                             |
       +--------------------------------+                |
       |          Decode Instance       |                |
       |                                |                |
       |   Decode Worker(s)             |                |
       |   - Qwen-7B model copy         |                |
       |   - KV Cache reception         |                |
       |   - Autoregressive Decode      |                |
       +--------------------------------+                |
              |                                         |
              +--------------------+--------------------+
                                   |
                                   v
                             Client Response
```

角色绑定不可互换：

-   `node2` 只承担 Prefill。
-   `node1` 只承担 Decode。
-   不得将 Prefill 和 Decode 放到同一物理节点上运行。
-   每个 Instance 只能使用分配给它的两张指定 GPU，不得占用节点上的其他
    GPU。
-   如果实际节点 GPU 数量大于两张，必须通过设备可见性和 Ray
    资源约束确保隔离；不能仅依赖"启动时希望使用两张卡"。

------------------------------------------------------------------------

## 2. 开始工作前的强制规则

### 2.1 先检查，再修改

Agent 必须按本文档的阶段顺序工作：

1.  检查仓库、论文和已有脚本。
2.  只读检查两台服务器的硬件、软件和网络。
3.  提交架构梳理与改造计划。
4.  实施最小必要代码修改。
5.  完成模型、数据和部署准备。
6.  分阶段验证，最后进行端到端运行。

在完成仓库检查前，不得直接重写 DistServe
的核心模块；在完成硬件盘点前，不得写死 GPU ID、网络接口、IP、端口或 NCCL
环境变量。

### 2.2 远程访问与凭据

-   等待用户提供 `node1`、`node2` 的 SSH 连接信息后再连接。
-   不得猜测 IP、用户名、端口、密钥路径或 SSH Host 别名。
-   不得要求用户把密码、私钥、Token 写入仓库或 `AGENTS.md`。
-   凭据应通过用户指定的 SSH Agent、SSH
    config、环境变量或安全凭据方式提供。
-   不得将凭据打印到终端日志、实验日志或提交到 Git。

### 2.3 高影响操作必须先征求用户同意

以下操作在执行前必须说明目的、影响和回滚方式，并取得用户同意：

-   安装或升级系统级软件、驱动、CUDA、NCCL、网络驱动。
-   使用 `sudo` 修改系统配置。
-   修改防火墙、路由、网卡、RDMA、NCCL 或 SSH 配置。
-   重启服务器、网络服务或 GPU 服务。
-   删除、覆盖或移动用户已有数据、模型和实验结果。
-   启动长时间下载、大规模数据处理、长时间模型推理或高并发压测。
-   占用用户未明确分配的 GPU、CPU、磁盘或网络资源。

优先使用项目独立的 Conda/venv 环境和用户目录，不要擅自污染系统 Python
环境。

### 2.4 不确定时不得猜测

如果代码、硬件或数据集信息不足：

-   先输出已确认事实、未知项和需要执行的检查。
-   对影响正确性的关键歧义，向用户提出明确问题。
-   不得用"通常如此"代替对当前仓库或服务器的实际检查。
-   不得为了让脚本运行而隐藏错误或跳过关键校验。

------------------------------------------------------------------------

## 3. 阶段 A：DistServe 仓库与论文代码审查

在改代码前，先阅读项目根目录说明、依赖文件、启动脚本和核心模块，并结合
DistServe 原论文理解原始设计。

重点审查以下模块和调用链：

  --------------------------------------------------------------------------------------------
  检查对象                            需要回答的问题
  ----------------------------------- --------------------------------------------------------
  Controller / Frontend               请求从哪里进入？使用什么 API？Controller
                                      如何维护请求状态？

  Orchestration                       谁负责启动和管理 Prefill/Decode Instance？

  Instance                            代码中 Instance 的定义是什么？是否对应一份完整模型权重？

  Worker                              Worker 如何创建？是否由 Ray Actor 管理？一个 Worker
                                      对应什么 GPU 资源？

  Placement                           原 Placement 算法如何分配 GPU、TP、PP 和
                                      Instance？能否限制为固定双节点拓扑？

  Scheduler                           Prefill 和 Decode 的队列、调度策略和负载选择方式是什么？

  KV Cache                            KV Cache 在 Prefill 端如何分配、保存、标识和释放？

  Transfer                            KV Cache 通过什么机制传输？是 NCCL、TCP、RDMA
                                      还是其他实现？

  Model executor                      当前代码支持哪些模型架构、Tokenizer、dtype 和并行策略？

  Startup / Shutdown                  各服务的启动顺序、依赖关系、健康检查和清理方式是什么？
  --------------------------------------------------------------------------------------------

至少画出当前仓库的实际调用链：

``` text
Request
  -> Frontend / Controller
  -> Prefill scheduling
  -> Prefill Worker(s)
  -> KV Cache creation
  -> KV Cache transfer
  -> Decode scheduling
  -> Decode Worker(s)
  -> Token generation
  -> Response
```

上图是目标逻辑流程，不代表当前代码已经按此结构实现。Agent
必须根据代码补充真实模块名、函数名、进程边界和通信接口。

### 3.1 论文实现中需要保留的机制

阅读论文时重点核对：

-   DistServe 将 Prefill 和 Decode 分配到不同 GPU
    资源上，两个阶段分别维护模型权重副本。
-   一个 Instance 是管理一份完整模型权重副本的资源单元，一个 Instance
    可以使用多张 GPU。
-   论文运行时由 Controller 接收请求，先分派给 Prefill
    Instance，再分派给 Decode Instance。
-   论文使用 FCFS 作为基础调度策略，并根据队列负载选择实例。
-   论文提出 Decode 端按需拉取 KV Cache（pull），而不是无条件由 Prefill
    端推送，以缓解突发请求造成的 Decode 端内存压力。
-   论文实现使用 NCCL 进行跨节点 GPU 通信、使用异步 CUDA
    内存拷贝处理节点内通信，并使用 Ray Actor 实现 GPU Worker。

上述内容来自 DistServe
论文的运行时架构、在线调度和实现部分。当前仓库若与论文版本不同，必须记录差异，以仓库实际实现为准。

### 3.2 阶段 A 的交付物

完成审查后，先形成一份简短的代码审查报告，至少包括：

1.  仓库版本、分支和 commit。
2.  核心模块及其文件路径。
3.  Instance、Worker、Ray Actor 和 GPU 的真实映射关系。
4.  当前支持的模型、并行方式和 KV Cache 传输方式。
5.  当前启动流程和配置入口。
6.  适配两节点固定角色部署时必须修改的代码。
7.  可以复用、不应该修改的模块。
8.  尚待用户提供的信息。

------------------------------------------------------------------------

## 4. 阶段 B：两台服务器硬件、软件与网络盘点

取得用户提供的 SSH
连接方式后，先执行只读检查。不要一连接就安装依赖或启动模型。

### 4.1 GPU 与计算资源

在两台节点分别检查并记录：

-   GPU 型号、数量、物理 GPU ID、显存容量和当前显存占用。
-   GPU 驱动版本、CUDA Driver 版本、可用 CUDA Runtime。
-   GPU 之间的拓扑关系：PCIe、NVLink（如果存在）、NUMA 亲和性。
-   CPU 型号、核心数、内存容量。
-   磁盘剩余空间、模型缓存目录和数据目录。
-   是否存在其他用户进程占用 GPU。

常见只读检查命令（具体命令是否可用需先确认）：

``` bash
nvidia-smi
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
lscpu
free -h
df -h
```

**不要预设 A5000/A6000 的实际数量、显存、GPU ID 或 GPU 拓扑。**
以服务器输出为准。

### 4.2 软件环境

分别记录：

-   OS 发行版和内核版本。
-   Python 版本。
-   PyTorch 版本及其 CUDA 构建版本。
-   CUDA Toolkit（若安装）、NCCL 版本。
-   Ray 版本。
-   DistServe 所需的其他依赖、编译器和自定义 CUDA/C++ 扩展。
-   当前仓库要求的版本与服务器现有版本是否兼容。

不要因为某个依赖版本较旧就直接升级。先判断原始 DistServe
代码依赖，再决定使用独立环境、补丁或兼容性修改。

### 4.3 网络与跨节点通信

由于 Prefill 和 Decode 位于不同物理节点，KV Cache
必须跨节点传输。必须检查：

-   两台服务器之间的 IP 连通性和路由。
-   可用网卡、网卡速率、MTU、网卡驱动。
-   是否存在 InfiniBand 或 RoCE/RDMA 设备。
-   实际可用的跨节点带宽和延迟。
-   NCCL 是否能选择正确的网络接口。
-   相关端口是否可达，是否受到防火墙或安全策略限制。
-   Ray 节点发现、Worker 注册和跨节点通信是否可用。

不要假设两台服务器处于同一高速网络，也不要假设 NCCL 一定走
RDMA。必须通过实际检查和通信测试确认。

如果原实现使用 NCCL 进行跨节点 KV Cache
传输，应优先复用原机制。若当前网络不支持或性能不满足要求：

1.  先定位是网络、NCCL 配置还是代码兼容问题。
2.  给出诊断结果和候选方案。
3.  只有在用户同意后才修改网络或更换传输后端。
4.  不得静默将 NCCL/RDMA 改成 TCP，并仍声称使用原始传输路径。

### 4.4 GPU 资源隔离

盘点完成后，为每台机器确定两张实际分配给本项目的 GPU：

-   `node1`：两张 A5000，供 Decode Instance 使用。
-   `node2`：两张 A6000，供 Prefill Instance 使用。

将实际物理 GPU ID 写入独立配置文件，不要硬编码在 Python
源码中。启动时使用经过验证的 `CUDA_VISIBLE_DEVICES` 和 Ray GPU
资源配置，确保：

-   Prefill Worker 只能使用 `node2` 分配的两张 A6000。
-   Decode Worker 只能使用 `node1` 分配的两张 A5000。
-   Controller 或其他辅助进程不意外占用这四张 GPU。
-   Ray 报告的 GPU 资源数量与实际可见设备一致。

------------------------------------------------------------------------

## 5. 阶段 C：适配 DistServe 到固定双节点异构硬件

### 5.1 固定 Placement

本项目采用固定 Placement：

``` yaml
cluster:
  node1:
    role: decode
    gpu_count: 2
    gpu_type: A5000

  node2:
    role: prefill
    gpu_count: 2
    gpu_type: A6000

instances:
  prefill:
    node: node2
    gpu_count: 2
    enabled: true

  decode:
    node: node1
    gpu_count: 2
    enabled: true
```

以上是逻辑配置示意，不是可以直接运行的完整配置。实际配置必须补充仓库要求的字段、GPU
ID、网络地址、端口和并行参数。

要求：

-   一个 Prefill Instance，固定在 `node2`。
-   一个 Decode Instance，固定在 `node1`。
-   每个 Instance 各使用两张 GPU。
-   不创建额外 Prefill/Decode Replica，除非后续用户明确要求。
-   不让 Placement 算法将实例迁移到其他节点。
-   如果原 Placement 模块不支持固定
    Placement，优先增加一个配置驱动的固定 Placement
    模式，而不是重写整个调度系统。
-   论文中的自动资源搜索和当前实验的固定资源分配要明确区分。当前固定配置是实验约束，不应伪装成原论文自动搜索得出的最优结果。

### 5.2 并行策略

DistServe 论文中涉及两类并行：

-   **Intra-op parallelism**：对应算子内并行，典型实现为 Tensor
    Parallelism（TP）。
-   **Inter-op parallelism**：对应算子/层之间的流水线并行，典型实现为
    Pipeline Parallelism（PP）。

每个 Instance 有两张 GPU，但不能仅凭 GPU 数量就决定 TP/PP。Agent 必须：

1.  检查当前 DistServe 代码对 Qwen-7B 的 TP/PP 支持。
2.  检查 Qwen-7B 模型结构、Attention/KV Head 配置和模型权重加载方式。
3.  检查 A5000、A6000 的实际显存与 GPU 互联拓扑。
4.  检查 KV Cache 在 Prefill 和 Decode 两端的分片布局是否兼容。
5.  通过小规模加载和实际通信验证配置。

可将 `TP=2, PP=1` 作为优先评估的候选配置，因为每个阶段分配两张
GPU；但**未经代码检查、显存验证和 KV Cache
兼容性验证，不得将其视为已经确定的配置**。

不要为了凑齐两张 GPU 而将 TP 跨节点拆分。Prefill 和 Decode
本来就位于不同节点，跨阶段 KV Cache 传输与模型内部 TP
通信是两回事。除非原实现明确支持、网络测试通过且确有必要，否则每个
Instance 的模型并行只使用其所在节点内部的两张 GPU。

### 5.3 异构 GPU 适配

A5000 和 A6000 分属不同硬件节点。Agent 必须检查：

-   两端 CUDA Driver/Runtime、PyTorch、NCCL 和自定义算子是否兼容。
-   两端是否使用相同的模型配置、权重 revision、Tokenizer 和 KV Cache
    dtype。
-   两端 TP/PP 配置是否能产生兼容的 KV Cache 布局。
-   跨节点传输是否正确处理不同 GPU 型号带来的执行时间差异。
-   日志和性能数据是否记录每个阶段实际使用的 GPU 型号。

不要把两个节点当成同构机器，也不要因为两边 GPU
数量相同就假定执行时间和显存容量相同。

### 5.4 KV Cache 传输正确性

这是 PD 分离能否正确运行的核心。必须沿用并验证 DistServe
原有传输路径，不能只验证网络能 ping 通。

至少核对以下信息在两端是否一致或能够正确映射：

-   Model ID、模型 revision、模型配置和层数。
-   Attention Head / KV Head 数量及排列。
-   TP/PP 分片方式和 Layer-to-Worker 映射。
-   KV Cache dtype、张量 shape、stride、内存布局和设备位置。
-   Request ID、序列长度、Prompt Token 数、当前 Token 位置。
-   Prefill 产生的首个输出 Token（如果原实现将其与 KV Cache 一起交给
    Decode）。
-   KV Cache 生命周期、释放时机和并发请求隔离。

验证要求：

1.  Prefill 完成前，Decode 不得读取尚未完成的 KV Cache。
2.  Decode 必须拿到与当前请求对应的 KV Cache，不能串请求。
3.  KV Cache 传输完成后再进入对应 Decode 计算。
4.  多请求并发时，Request ID 与 KV Cache 的映射必须保持正确。
5.  传输失败、超时或请求取消时，状态和显存必须能正确清理。
6.  通过小批量请求验证端到端生成结果，而不是仅验证发送和接收函数返回成功。

如果原实现使用 Decode 端 pull KV Cache
的方式，应优先保留该语义，不要未经分析改成 Prefill 主动 push。

### 5.5 修改原则

-   先找出现有配置入口，优先配置化，不要把节点地址、GPU
    ID、端口写死在代码中。
-   只修改适配双节点异构环境所必需的模块。
-   尽量保留 Controller、调度策略、请求状态管理和 KV Cache 传输协议。
-   不要额外引入磁盘级 KV
    Cache、抢占、缓存压缩或其他论文外优化，除非用户明确要求。
-   每一项偏离原始 DistServe
    的修改都要记录：修改文件、原因、行为变化、配置开关、测试方式和结果。
-   对原始实现无法支持的能力，先报告限制，再提出最小改造方案。

------------------------------------------------------------------------

## 6. 阶段 D：Qwen-7B 模型准备

### 6.1 模型下载位置

模型权重在服务器端下载，不在用户本地下载后再上传，除非用户后续另行指定。

由于 Prefill 和 Decode 是两个独立
Instance，必须保证两个节点都能访问完整模型权重：

-   `node2` 的 Prefill Instance 能加载完整 Qwen-7B 模型。
-   `node1` 的 Decode Instance 能加载同一版本的完整 Qwen-7B 模型。

如果两个节点没有共享文件系统，则需要分别下载或通过用户认可的方式传输模型文件。不得默认存在共享存储。

### 6.2 下载前确认

下载前检查并确认：

-   Hugging Face Model ID。
-   模型变体：Base、Chat 或其他版本。
-   Revision / commit。
-   Tokenizer 和模型配置。
-   模型是否需要 `trust_remote_code`。
-   当前 DistServe 是否支持该模型架构。
-   权重格式和加载依赖。
-   模型最大上下文长度。
-   当前实现支持的 dtype。

`Qwen-7B` 是目标模型系列描述，不足以让 Agent
静默选择其他代际或指令微调版本。若仓库 ID 或变体不明确，先向用户确认。

### 6.3 模型适配与精度

DistServe 论文实验使用 FP16，但当前 Qwen-7B
代码路径和硬件环境仍需实际验证。

要求：

-   优先检查原 DistServe
    的模型加载实现，不要直接替换成另一个推理框架来绕过模型适配。
-   如果当前代码不支持 Qwen
    架构，先定位模型注册、Attention、RoPE、Tokenizer、权重命名和 KV
    Cache 接口的缺失部分。
-   FP16 可作为优先验证候选；是否最终使用
    FP16/BF16，以代码支持、模型要求和实际显存验证为准。
-   Prefill 与 Decode 两端必须使用相同模型权重版本和兼容的 KV Cache
    dtype。
-   不得未经记录地使用量化模型或不同精度权重。
-   加载成功后记录模型 ID、revision、dtype、最大上下文长度、权重目录和
    GPU 显存占用。

### 6.4 模型验收

-   两端分别能加载模型。
-   两端模型配置和权重版本一致。
-   GPU 使用范围符合固定拓扑。
-   没有 OOM。
-   可以在各自 Instance 内完成必要的最小模型计算。
-   Prefill 和 Decode 的 KV Cache 格式经过检查并能够互通。

------------------------------------------------------------------------

## 7. 阶段 E：WildChat-1M 本地数据准备与泊松到达时间生成

### 7.1 数据处理位置

严格遵守以下数据流：

``` text
用户本地机器
  |
  +-- 下载 allenai/WildChat-1M
  |
  +-- 检查 Dataset schema / split / license
  |
  +-- 清洗、筛选和构造请求
  |
  +-- 统计输入/输出 Token 长度
  |
  +-- 生成 Poisson arrival timestamps
  |
  +-- 保存 processed workload + manifest
  |
  +-- 上传处理后的文件到服务器
  |
  v
服务器读取 workload 并运行实验
```

-   数据集原始下载和预处理在用户本地完成。
-   服务器只接收处理后的 workload 文件和必要的 manifest。
-   不要在服务器上重新下载 WildChat-1M。
-   不得将用户本地绝对路径写死到仓库脚本或配置中。
-   如果 Agent
    无法直接访问用户本地机器，应提供可复现的本地脚本和命令，由用户执行或在用户授权的本地环境执行。

### 7.2 数据集检查

开始处理前，检查 Hugging Face 数据集的：

-   License 和使用限制。
-   Dataset card、数据版本和 revision。
-   Split 名称和数据规模。
-   实际字段名、字段类型和嵌套结构。
-   对话消息的 role、content、时间或其他元数据字段。
-   空值、异常样本、重复样本和超长样本比例。

不得假设字段一定叫 `conversation`、`messages`、`user` 或
`assistant`。必须先读取真实 schema，再实现解析器。

### 7.3 请求构造策略

WildChat-1M 是对话数据集，不应直接把原始 JSON 行当作推理请求。Agent
必须将对话转换为 DistServe workload 所需的请求记录。

先检查数据结构，并形成请求抽取策略报告。建议优先评估以下默认策略：

-   一条有效对话记录映射为一个推理请求。
-   以最后一个有效 User 消息作为当前请求的输入终点。
-   将该 User 消息之前的有效对话历史作为上下文，按 Qwen-7B 对应的 Chat
    Template 构造 Prompt。
-   将其后对应的 Assistant
    回复作为参考输出，用于统计输出长度或质量分析。
-   如果找不到有效 User 输入或对应 Assistant
    回复，则按明确的过滤规则跳过。
-   对
    tool、system、拒答、空消息、多模态内容等特殊情况，先统计数量，再按用户确认的规则处理。

以上是建议的默认方案，不是对 WildChat 数据 schema
的预设。如果实际数据结构不适用，Agent
必须先报告并提出调整方案，不得静默改变请求语义。

每个请求至少保留：

-   `request_id`：全局唯一且可复现。
-   `source_record_id`：原始数据记录标识（若数据集提供）。
-   `prompt`：最终构造的输入。
-   `reference_response`：可选的参考回复。
-   `input_token_length`：使用目标 Qwen Tokenizer 计算。
-   `output_token_length`：参考回复的 Token 数。
-   `max_new_tokens`：本次服务请求的生成长度上限。
-   `arrival_time_s`：相对 workload 起点的到达时间，单位为秒。
-   `interarrival_time_s`：与上一个请求的时间间隔，单位为秒。
-   `arrival_rate_rps`：该 trace 使用的目标请求到达率。
-   `dataset_revision`、`preprocess_version` 和随机种子等复现信息。

注意：参考回复不等于模型必须生成的目标文本。性能测试时通常用参考回复长度确定
`max_new_tokens`；是否保存或使用参考文本做质量评估，应与性能测试逻辑分开。

### 7.4 泊松到达过程

DistServe 论文在实验设置中指出，所用数据集没有请求时间戳，因此通过
Poisson distribution
生成请求到达时间，并使用不同请求率进行实验。本项目采用相同的 workload
生成思路。

对于目标请求到达率
(`\lambda`{=tex})（单位：requests/second），请求间隔应满足：

\[ `\Delta `{=tex}t_i
`\sim `{=tex}`\operatorname{Exponential}`{=tex}(`\lambda`{=tex}) \]

等价地：

\[ `\Delta `{=tex}t_i = -`\frac{\ln U_i}{\lambda}`{=tex},
`\qquad `{=tex}U_i `\sim `{=tex}`\operatorname{Uniform}`{=tex}(0,1) \]

到达时间戳为：

\[ t_0 = 0,`\qquad`{=tex} t_i = t\_{i-1} + `\Delta `{=tex}t_i \]

实现要求：

-   `lambda` 必须是可配置参数，单位明确为 req/s。
-   使用独立、明确记录的随机种子，确保同一配置可复现。
-   使用指数分布采样 inter-arrival time，不得用固定时间间隔 `1/lambda`
    冒充泊松过程。
-   到达时间必须严格单调递增。
-   `arrival_time_s` 使用相对 workload
    开始时刻的秒数，不依赖用户机器或服务器的绝对时钟。
-   对多组请求率生成多份 trace，并在文件名或 manifest
    中记录请求率和随机种子。
-   对每组 trace 记录请求数量、总持续时间、平均实际到达率和
    inter-arrival time 统计。
-   允许配置样本数、过滤条件和最大 Prompt/Output 长度。
-   不要为了让 trace
    时长整齐而强行修改采样结果；如需截断，应记录截断规则。
-   如果需要公平比较多个请求率，可使用同一组基础随机数生成可缩放
    trace；具体策略必须写入 manifest。

重要说明：WildChat-1M
不提供可直接代表真实在线服务到达过程的请求时间戳。本项目生成的 Poisson
时间戳是**合成到达过程**，不是恢复或推断 WildChat 用户的真实访问时间。

### 7.5 Token 长度和数据过滤

-   使用服务器实际部署的同一模型 Tokenizer，或在本地使用完全相同版本的
    Tokenizer。
-   记录 Tokenizer 的 Model ID 和 revision。
-   过滤规则必须配置化，例如空
    Prompt、无有效回复、超出上下文长度、异常字符和不支持的内容类型。
-   对长 Prompt、长回复应明确采用跳过、截断还是单独分组，不得静默截断。
-   统计过滤前后数据量、Prompt 长度分布、Output 长度分布和过滤原因。
-   对采样、洗牌和过滤使用确定性随机种子。
-   原始下载数据不得被覆盖或修改。

### 7.6 本地数据处理产物

建议至少产出：

``` text
workload/
  raw/                         # 原始下载数据，不覆盖
  processed/
    wildchat_poisson_*.jsonl   # 处理后的请求 trace
    manifest.json              # 数据集版本、过滤规则、lambda、seed 等
    statistics.json            # 样本数、长度分布、到达间隔统计
  scripts/
    download_wildchat.py
    preprocess_wildchat.py
    generate_poisson_trace.py
  configs/
    workload.yaml
```

实际目录可以适配仓库现有结构，但必须保留原始数据、处理脚本、配置和统计信息之间的对应关系。

### 7.7 上传到服务器

-   上传处理后的 workload 和 manifest，不要求上传原始 1M 数据集。
-   上传前计算文件大小和 SHA256。
-   上传后在服务器端重新计算 SHA256，确认文件一致。
-   上传目标目录由用户确认，不得覆盖服务器已有文件。
-   不得将本地文件路径、用户名、密钥或访问 Token 写入 manifest。

------------------------------------------------------------------------

## 8. 阶段 F：部署配置与启动流程

### 8.1 配置分层

节点、GPU、模型、网络和 workload
参数应放在配置文件或环境变量中。不要将实验参数散落在源代码中。

建议配置至少包含：

``` yaml
cluster:
  node1:
    role: decode
    address: "<由用户提供>"
    gpu_ids: ["<盘点后确认>", "<盘点后确认>"]

  node2:
    role: prefill
    address: "<由用户提供>"
    gpu_ids: ["<盘点后确认>", "<盘点后确认>"]

model:
  model_id: "<确认后的 Qwen-7B Hugging Face ID>"
  revision: "<确认后的 revision>"
  dtype: "<验证后的 dtype>"
  max_model_len: "<检查模型配置后确定>"

instances:
  prefill:
    node: node2
    gpu_count: 2
    tp: "<验证后确定>"
    pp: "<验证后确定>"

  decode:
    node: node1
    gpu_count: 2
    tp: "<验证后确定>"
    pp: "<验证后确定>"

workload:
  path: "<服务器上的 processed workload 路径>"
  arrival_rate_rps: "<实验配置>"
  seed: "<实验配置>"
```

这只是配置结构示意。实际字段必须与当前仓库配置系统兼容。

### 8.2 启动顺序

Agent 必须先从代码确认实际依赖关系，再生成启动脚本。预期逻辑顺序如下：

1.  检查两个节点的 GPU 和网络状态。
2.  启动或连接 Ray / 分布式执行环境（如果当前 DistServe 实现需要）。
3.  启动 Decode 侧 Worker 和 Decode Instance。
4.  启动 Prefill 侧 Worker 和 Prefill Instance。
5.  启动 Controller / Frontend / Orchestration。
6.  执行健康检查，确认两个 Instance 注册成功且 GPU 绑定正确。
7.  提交一个小请求，验证 Prefill、KV Cache Transfer 和 Decode 完整链路。
8.  只有小请求通过后，才启动正式 workload。

如果当前代码要求不同的启动顺序，以实际依赖为准，并在 README 中解释。

### 8.3 启动脚本要求

提供可重复执行的脚本或命令，至少包括：

-   环境检查。
-   Prefill Instance 启动。
-   Decode Instance 启动。
-   Controller 启动。
-   服务状态检查。
-   日志收集。
-   正常停止和资源清理。

脚本必须：

-   从配置读取节点和 GPU 信息。
-   在启动前检查 GPU 是否已被其他进程占用。
-   检查端口冲突。
-   检查模型目录和 workload 文件是否存在。
-   检查 GPU 可见设备数量与配置是否一致。
-   失败时返回非零退出码，并打印可定位的错误信息。
-   不得通过忽略异常、吞掉返回码或后台启动后不检查状态来伪装成功。

------------------------------------------------------------------------

## 9. 阶段 G：分层联调与验收

必须按以下顺序验证，不得直接用完整 WildChat workload 作为第一次测试。

### G1. 单节点环境检查

-   两端 Python、PyTorch、CUDA、NCCL、Ray 版本符合已确认的兼容方案。
-   两端都能看到各自指定的两张 GPU。
-   GPU ID 和 `CUDA_VISIBLE_DEVICES` 映射正确。
-   模型目录可访问。
-   无意外占用其他 GPU。

### G2. Ray / Worker 检查

如果当前实现使用 Ray：

-   Ray 节点注册成功。
-   Worker Actor 能够在指定 Node 上创建。
-   Prefill Worker 只分配到 `node2` 的两张 A6000。
-   Decode Worker 只分配到 `node1` 的两张 A5000。
-   Ray 报告的资源数量和实际设备数量一致。
-   Worker 异常退出能够被发现，日志能定位到具体节点和 Worker。

如果当前实现没有使用 Ray，不要为了符合文档而强行引入
Ray；应记录当前执行模型。

### G3. 跨节点通信检查

-   两端网络可达。
-   NCCL 或原实现使用的传输后端初始化成功。
-   能完成最小跨节点 GPU 通信测试。
-   记录使用的网卡、协议、带宽和延迟。
-   不得仅以 ping 成功作为 KV Cache 通信通过的依据。

### G4. KV Cache 专项检查

使用最小请求验证：

1.  Prefill 端收到 Prompt。
2.  Prefill 端完成计算并生成 KV Cache。
3.  Decode 端按原设计获取对应 KV Cache 和必要请求状态。
4.  传输数据大小、Tensor shape、dtype、Layer/Head 映射符合预期。
5.  Decode 端在 KV Cache 可用后开始生成。
6.  请求结束后两端相关状态正确释放。

必要时增加校验日志、张量元数据检查或测试用
checksum，但不要在正式性能测量路径中长期保留高开销调试逻辑。

### G5. 端到端功能检查

依次测试：

-   单个短 Prompt。
-   单个较长 Prompt。
-   多个顺序请求。
-   多个并发请求。
-   不同 Prompt 长度。
-   不同 `max_new_tokens`。
-   请求取消或异常时的清理行为。

检查：

-   `node2` 只执行 Prefill。
-   `node1` 只执行 Decode。
-   Decode 没有重复执行 Prefill。
-   请求不会串线，KV Cache 不会错配。
-   返回结果非空且生成过程正常。
-   TTFT、TPOT 和端到端延迟能够正确记录。
-   没有 GPU OOM、Worker 崩溃或 KV Cache 泄漏。

### G6. 正式 workload 前检查

正式压测前必须确认：

-   workload 文件及 manifest 的 SHA256 一致。
-   到达率、随机种子和请求数与实验配置一致。
-   到达时间严格递增，间隔分布符合指数分布的预期。
-   请求输入长度和生成长度分布已统计。
-   服务器剩余磁盘、GPU 显存和网络状态满足实验要求。
-   正式实验不会覆盖已有日志和结果。

------------------------------------------------------------------------

## 10. 实验记录与可复现性

每次运行必须记录：

### 环境信息

-   DistServe 仓库 URL、branch、commit。
-   本项目代码改动 commit 或 patch。
-   两台节点的 GPU 型号、GPU ID、显存。
-   Driver、CUDA、PyTorch、NCCL、Ray、Python 版本。
-   网络接口、协议、带宽和 NCCL 配置。
-   模型 ID、revision、dtype、Tokenizer revision。

### 部署信息

-   Prefill Instance 所在节点、GPU ID、TP、PP。
-   Decode Instance 所在节点、GPU ID、TP、PP。
-   Worker 数量和进程映射。
-   Controller、Ray 和服务启动参数。
-   KV Cache 传输后端和配置。

### Workload 信息

-   WildChat-1M dataset revision。
-   本地预处理脚本版本和配置。
-   过滤规则及过滤后样本数。
-   Poisson 到达率（req/s）、随机种子和 trace 文件 SHA256。
-   Prompt / Output Token 长度统计。
-   Workload 总持续时间。

### 运行信息

-   每个服务的启动时间和退出状态。
-   标准输出、错误输出和服务日志。
-   TTFT、TPOT、E2E latency。
-   Request throughput、Token throughput。
-   SLO attainment（如果配置了 TTFT/TPOT SLO）。
-   Prefill queue、Decode queue 和 KV Cache transfer 时间。
-   GPU 利用率、显存占用、网络带宽。
-   失败请求数量及原因。

每次运行使用独立的结果目录，不得覆盖已有实验结果。

------------------------------------------------------------------------

## 11. 建议的项目产物

在不破坏当前仓库结构的前提下，尽量形成以下可维护产物：

``` text
project/
  AGENTS.md
  README.md
  configs/
    cluster.yaml
    model.yaml
    workload.yaml
    deployment.yaml
  scripts/
    inspect_nodes.sh
    start_prefill.sh
    start_decode.sh
    start_controller.sh
    check_deployment.sh
    stop_services.sh
  workload/
    scripts/
      download_wildchat.py
      preprocess_wildchat.py
      generate_poisson_trace.py
    configs/
      wildchat.yaml
    processed/
      manifest.json
      statistics.json
  docs/
    architecture.md
    deployment_notes.md
    compatibility_report.md
    deviations_from_distserve.md
  logs/
  results/
```

如果仓库已经有相应目录或脚本，应优先复用，不要为了匹配示例目录而重复创建同类文件。

------------------------------------------------------------------------

## 12. 完成标准（Definition of Done）

只有同时满足以下条件，才可以认为"DistServe 双节点部署准备完成"：

-   [ ] 已审查 DistServe 原始代码和论文，明确当前代码的
    Instance、Worker、调度和 KV Cache 实现。
-   [ ] 已完成 `node1` 和 `node2` 的硬件、软件、GPU 和网络盘点。
-   [ ] 两个 Instance 固定绑定到指定节点和指定的两张 GPU。
-   [ ] Prefill Instance 在 `node2` 的两张 A6000 上运行。
-   [ ] Decode Instance 在 `node1` 的两张 A5000 上运行。
-   [ ] Qwen-7B 的确切模型 ID、revision、dtype 和加载方式已经确认。
-   [ ] 两端分别加载兼容的完整模型权重。
-   [ ] KV Cache 跨节点传输通过实际请求验证。
-   [ ] Controller 能够完成请求派发、阶段切换和结果返回。
-   [ ] WildChat-1M 已在本地完成下载、检查和预处理。
-   [ ] 泊松到达时间由指数分布的 inter-arrival time
    累积生成，并有可复现的随机种子。
-   [ ] 处理后的 workload 已上传服务器，文件校验通过。
-   [ ] 单请求、顺序请求和并发请求端到端测试通过。
-   [ ] 启动、停止、健康检查和日志收集脚本可重复执行。
-   [ ] 所有偏离原始 DistServe 的改动均有记录。
-   [ ] 已生成环境清单、部署说明和端到端验收报告。

如果任一项未完成，应明确标记为未完成，并说明阻塞原因。不得仅凭服务进程启动成功就宣布复现完成。

------------------------------------------------------------------------

## 13. Agent 每个阶段结束时的汇报格式

每完成一个阶段，按以下格式汇报：

1.  **已完成**：实际执行的检查、修改或测试。
2.  **关键发现**：代码结构、硬件信息、兼容性和运行结果。
3.  **修改记录**：文件路径、修改原因、对原 DistServe 语义的影响。
4.  **验证结果**：命令、关键输出和通过/失败结论。
5.  **待确认事项**：需要用户提供的 SSH、模型版本、网络信息或实验参数。
6.  **下一步计划**：下一阶段准备做什么，以及是否需要用户授权。

不要只报告"已经适配""可以运行"或"测试通过"；必须给出能够复核的证据。

------------------------------------------------------------------------

## 14. DistServe 原论文中的相关依据

本项目适配时应优先对照 DistServe 原论文：

-   **§2.3（约第 4 页）**：解释 Prefill/Decode 分离的动机，并定义
    Instance 为管理一份完整模型权重副本的资源单元；一个 Instance
    可以使用多张 GPU。
-   **§4.3（约第 8 页）**：描述在线调度。请求先进入 Controller，再派发到
    Prefill Instance，随后派发到 Decode Instance；论文还介绍了通过
    Decode 端 pull KV Cache 缓解突发流量。
-   **§5（约第 8--9 页）**：描述 DistServe 实现，包括 Placement、RESTful
    API Frontend、Orchestration Layer、并行执行引擎；论文实现使用 NCCL
    进行跨节点 GPU 通信、异步 CudaMemcpy 处理节点内通信，并使用 Ray
    Actor 实现 GPU Worker。
-   **§6.1（约第 9 页）**：论文实验使用
    FP16；由于所用数据集没有请求时间戳，作者使用 Poisson distribution
    生成不同请求率下的请求到达时间。
-   **Appendix B（约第 18 页）**：列出论文端到端实验中不同模型和
    workload 对应的 Prefill/Decode TP、PP 配置。该表是 OPT
    模型实验配置，不应未经验证直接套用到 Qwen-7B。

若论文描述与当前仓库实现不一致，必须在
`docs/deviations_from_distserve.md`
中列明差异，不得静默混用不同版本的设计。
