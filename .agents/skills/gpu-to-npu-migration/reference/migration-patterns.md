# GPU → NPU 迁移模式

## 官方迁移方法

华为昇腾支持三种迁移方法。**推荐方法 1。**

### 方法 1：自动迁移（transfer_to_npu）

添加一行 import — 所有 `torch.cuda` 调用在运行时自动替换为 `torch.npu`：

```python
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

**自动替换内容（无需修改代码）：**

| GPU（原代码） | NPU（运行时自动替换） |
|-----|-----|
| `torch.cuda.set_device(n)` | `torch.npu.set_device(n)` |
| `torch.cuda.is_available()` | `torch.npu.is_available()` → 返回 True |
| `torch.cuda.device_count()` | `torch.npu.device_count()` |
| `torch.cuda.current_device()` | `torch.npu.current_device()` |
| `torch.cuda.empty_cache()` | `torch.npu.empty_cache()` |
| `torch.cuda.synchronize()` | `torch.npu.synchronize()` |
| `torch.cuda.memory_allocated()` | `torch.npu.memory_allocated()` |
| `torch.cuda.max_memory_allocated()` | `torch.npu.max_memory_allocated()` |
| `torch.cuda.amp.autocast()` | `torch.npu.amp.autocast()` |
| `torch.cuda.amp.GradScaler()` | `torch.npu.amp.GradScaler()` |
| `torch.cuda.Stream()` | `torch.npu.Stream()` |
| `torch.cuda.Event()` | `torch.npu.Event()` |
| `torch.cuda.manual_seed()` | `torch.npu.manual_seed()` |
| `model.cuda()` | `model.npu()` |
| `tensor.cuda()` | `tensor.npu()` |
| `torch.device("cuda")` | `torch.device("npu")` |
| `torch.device("cuda:0")` | `torch.device("npu:0")` |
| `backend="nccl"` | `backend="hccl"` |
| `torch.cuda.DoubleTensor` | `torch.npu.FloatTensor`（NPU 不支持 double） |

**限制：**
- 与 `torch.jit.script` 冲突（使用动态 monkey-patching）
- 无法替换算子级调用如 `F.scaled_dot_product_attention`
- 无法替换 `ProfilerActivity.CUDA` 枚举值

### 方法 2：工具迁移（ms_fmk_transplt）

预转换脚本并生成迁移报告：
```bash
# 工具位于 CANN 安装路径
{CANN_PATH}/ascend-toolkit/latest/tools/ms_fmk_transplt/
```

### 方法 3：手动迁移

逐个替换 API。当方法 1-2 无法覆盖时使用。

## 仍需手动修改的内容

即使使用了 `transfer_to_npu`，以下内容仍需手动修改：

### 1. 注意力算子

```python
# 修改前（GPU）
output = F.scaled_dot_product_attention(q, k, v, ...)

# 修改后（NPU）
output = torch_npu.npu_fusion_attention(
    q, k, v, head_num,
    input_layout="BNSD",
    scale=1.0 / (head_dim ** 0.5),
    keep_prob=1.0 - dropout_p,
)[0]
```

详见 `npu-fusion-attention.md`。

### 2. Profiler Activity

```python
# 修改前
ProfilerActivity.CUDA
# 修改后
ProfilerActivity.NPU
```

## torch.accelerator API 兼容性

`torch.accelerator` API 在理论上是设备无关的，但存在版本差异。已验证：

| API | torch 2.6 | torch 2.9+ | 说明 |
|-----|-----------|------------|------|
| `torch.accelerator.is_available()` | 支持 | 支持 | 导入 torch_npu 后可用 |
| `torch.accelerator.device_count()` | 支持 | 支持 | 导入 torch_npu 后可用 |
| `torch.accelerator.current_accelerator()` | 支持 | 支持 | 导入 torch_npu 后返回 `"npu"` |
| `torch.accelerator.current_device_index()` | 支持 | 支持 | 返回当前设备索引（int） |
| `torch.accelerator.set_device_index(rank)` | 支持 | 支持 | 设置当前设备 — 用此替换 |
| `torch.accelerator.device_index(rank)` | **不支持** | 支持 | 上下文管理器，2.9 新增。2.6/2.7/2.8 不存在 |
| `torch.accelerator.empty_cache()` | **不支持** | 支持 | 内存相关函数 2.9 新增 |

**上游 FSDP2 nanoGPT (example.py) 使用的 API：**
- 第 34 行：`torch.accelerator.device_index(rank)` — torch ≤2.8 会报错 → 替换为 `torch.accelerator.set_device_index(rank)`（2.6+ 可用）或 `torch.npu.set_device(rank)`
- 第 41 行：`init_process_group(backend=backend, device_id=device)` — `device_id` 在 NPU 上报错 → 移除该参数

## 分布式训练

| 模式 | 修复方法 |
|------|---------|
| `init_process_group(backend="nccl")` | `transfer_to_npu` 自动处理 |
| `init_process_group(..., device_id=device)` | **移除 `device_id` 参数** — 仅支持 CUDA 设备 |
| `torch.distributed.get_default_backend_for_device()` | 正常工作 — 自动选择 HCCL |
| `torch.distributed.get_rank()` 返回字符串 | **转为 int：** `int(torch.distributed.get_rank())` — NPU 环境可能返回 `"0"` 而非 `0`，导致 `rank == 0` 为 False |

## NPU 上的 Checkpoint 保存问题

`torch.distributed.get_rank()` 在 NPU 上可能返回**字符串**（如 `"0"`）而非整数。这会导致 `rank == 0` 永远为 `False`（因为 Python 中 `"0" == 0` 为 `False`），rank 0 进程永远不会进入 checkpoint 保存分支。

**修复：** 将 `rank == 0` 替换为 `int(rank) == 0` 或 `int(torch.distributed.get_rank()) == 0`。
