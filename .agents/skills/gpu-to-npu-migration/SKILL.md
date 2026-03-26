---
name: gpu-to-npu-migration
description: 将 PyTorch GPU 训练代码迁移到华为昇腾 NPU。使用官方 transfer_to_npu 自动迁移，运行时自动替换所有 torch.cuda API；手动替换 F.scaled_dot_product_attention 为 torch_npu.npu_fusion_attention；处理 torch.accelerator API 兼容性问题。支持 torchrun 多卡分布式训练和 checkpoint 保存/加载验证。当用户要求迁移、适配、移植 GPU/CUDA 训练代码到 NPU，或提到"代码适配"、"GPU迁移NPU"、"迁移到昇腾"、"CUDA to NPU"、"模型迁移"时触发。
---

# GPU → NPU 模型迁移

将 PyTorch GPU 训练代码迁移到华为昇腾 NPU。

## 第 0 步：询问用户

执行任何操作前，先询问用户以下信息：

1. **源代码位置** — Git 仓库 URL 或本地路径。
2. **训练入口脚本** — 要运行哪个 `.py` 文件。
3. **NPU 卡数** — 使用多少张 NPU 卡进行分布式训练。
4. **容器/环境** — 是否已在 CANN 容器内？还是需要新建？

**如果用户未提供这些信息**，展示默认计划并等待确认：

> 我将执行以下迁移步骤：
>
> 1. 克隆 FSDP2 nanoGPT 代码：`https://github.com/pytorch/examples/tree/main/distributed/FSDP2`
> 2. 适配代码以兼容 NPU
> 3. 使用 `torchrun --nproc_per_node=2 example.py` 启动训练
> 4. 验证 checkpoint 保存和加载
>
> 是否继续？

**等待用户确认后再执行。**

## 工作流程

**关键：严格按照 1→2→3→4→5→6 的顺序执行。不要跳步。在第 2 步完成前，不要检查 torch_npu 或运行任何 Python import。干净容器中没有 torch 和 torch_npu — 这是正常的，将在第 2 步安装。**

### 第 1 步：克隆源代码

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf /tmp/pytorch-examples
git clone --depth 1 https://github.com/pytorch/examples.git /tmp/pytorch-examples
cd /tmp/pytorch-examples/distributed/FSDP2
```

### 第 2 步：安装依赖

**必须严格按此顺序 — 不要调换：**

```bash
# 2a. 先安装目标项目自身的依赖（这会安装 torch）
pip install -r requirements.txt

# 2b. torch 安装完成后，再安装版本匹配的 torch_npu
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
pip install torch_npu==${TORCH_VERSION} pyyaml

# 2c. 验证（仅在两者都安装完成后）
python3 -c "import torch, torch_npu; print(f'torch={torch.__version__} npu={torch_npu.__version__} devs={torch.npu.device_count()}')"
```

**torch_npu 版本必须与 torch 完全匹配**（如 torch 2.6.0 → torch_npu 2.6.0）。

### 第 3 步：适配代码

阅读项目中所有 `.py` 文件。以下修改相互独立，可并行执行：

**入口脚本（如 example.py）：**
- 添加 `import torch_npu` 和 `from torch_npu.contrib import transfer_to_npu`（插入到已有 import 之后）。`transfer_to_npu` 会在运行时自动替换所有 `torch.cuda.*` / `.cuda()` / `"cuda"` / `nccl→hccl`
- `torch.accelerator.device_index(rank)` → `torch.accelerator.set_device_index(rank)`（device_index 在 torch ≤2.8 不存在）
- `init_process_group(..., device_id=device)` → 移除 `device_id` 参数（仅支持 CUDA）
- `ProfilerActivity.CUDA` → `ProfilerActivity.NPU`（如果存在）

**模型文件（如 model.py）：**
- 添加 `import torch_npu`
- 替换 `F.scaled_dot_product_attention` → `torch_npu.npu_fusion_attention`，变量名需与实际代码匹配：
```python
output = torch_npu.npu_fusion_attention(
    query, key, value,
    head_num,                  # int: 注意力头数
    input_layout="BNSD",       # 匹配张量布局
    scale=1.0 / (head_dim ** 0.5),
    keep_prob=1.0 - dropout_p, # 取反：dropout_p → keep_prob
)[0]                           # 返回元组，取第一个元素
```
- 如果原代码使用 `is_causal=True`：添加 `sparse_mode=2`。详见 `reference/npu-fusion-attention.md`

**Checkpoint 文件（如 checkpoint.py）：**
- `torch.distributed.get_rank()` 在 NPU 上可能返回字符串 `"0"` 而非整数 `0`，导致 `rank == 0` 永远为 `False`。将所有 `rank == 0` 替换为 `int(rank) == 0`
- 确认 checkpoint 保存代码未被注释

### 第 4 步：启动训练

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
torchrun --nproc_per_node=<N> <training_script.py>
```

**如果 OOM：** 清理残留进程：`pkill -f "torchrun|python3"`
**如果 HCCL 错误：** `python3 -c "import torch, torch_npu; print(torch.npu.device_count())"`

### 第 5 步：验证 Checkpoint 保存和加载

1. 第一次训练后检查 checkpoint 文件是否存在
2. 再次运行训练 — 应该出现新的 checkpoint 文件
3. 对比第二次训练前后的 checkpoint 文件数量

### 第 6 步：报告结果

完成后，告知用户：

> **迁移完成。**
>
> **代码修改：**
> - 修改的文件：`<列表>`
> - 添加 import 的文件：`<入口脚本>`
> - 替换注意力算子的文件：`<文件>`
> - 其他修复：`<修复内容>`
>
> **训练产物：**
> - Checkpoint 路径：`<绝对路径>`
> - Checkpoint 文件：`<列表>`
>
> **验证结果：**
> - 第一次训练：exit code `<N>`
> - 恢复训练：exit code `<N>`，checkpoint 数量 `<之前>` → `<之后>`
>
> **源代码位置：** `<适配后的项目路径>`

## 故障排查

### "ModuleNotFoundError: No module named 'torch_npu'"
`pip install torch_npu==$(python3 -c "import torch; print(torch.__version__.split('+')[0])")`

### "No NPU devices available"
`source /usr/local/Ascend/ascend-toolkit/set_env.sh`

### HCCL 初始化失败
清理残留进程，确保有 ≥2 张 NPU 卡可用。

### "torch.accelerator has no attribute 'device_index'"
替换为 `torch.accelerator.set_device_index(rank)`。该 API 在 torch 2.9+ 才可用。

### "device_id parameter must be a cuda device"
从 `init_process_group()` 调用中移除 `device_id=device` 参数。

### "torch.jit.script" 报错
`transfer_to_npu` 使用的 monkey-patching 与 `torch.jit.script` 冲突。移除 JIT 装饰器或改用手动迁移。

### 训练完成后 checkpoint 文件未保存
`torch.distributed.get_rank()` 在 NPU 上可能返回字符串 `"0"`。`"0" == 0` 在 Python 中为 `False`，导致 rank 0 永远不会保存。修复：`int(torch.distributed.get_rank()) == 0`。

### 注意力算子输出结果错误
检查原代码是否使用 `is_causal=True`。如果是，需要添加 `sparse_mode=2`。详见 `reference/npu-fusion-attention.md`。

## 参考资料

- `reference/npu-fusion-attention.md` — 注意力算子完整 API 和参数映射
- `reference/migration-patterns.md` — transfer_to_npu 自动处理的内容 vs 需要手动修改的内容
- 官方文档：https://www.hiascend.com/document/detail/zh/canncommercial/850/devaids/migrationtools/atlasfmkt_16_0019.html
- 使用用户的语言回复
- 不要翻译：命令、文件路径、环境变量、包名、错误信息
