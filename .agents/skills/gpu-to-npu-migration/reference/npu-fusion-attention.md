# npu_fusion_attention API 参考

## 函数签名

```python
torch_npu.npu_fusion_attention(
    query,          # Tensor: BNSD 布局为 (B, N, S, D)
    key,            # Tensor: (B, N, S, D)
    value,          # Tensor: (B, N, S, D)
    head_num,       # int: 注意力头数
    input_layout,   # str: "BNSD" | "BSH" | "SBH" | "BSND" | "TND"
    scale=1.0/sqrt(head_dim),  # float: 注意力缩放因子
    atten_mask=None,     # Tensor 或 None: 注意力掩码
    keep_prob=1.0,       # float: 1.0 - dropout_p
    sparse_mode=0,       # int: 0=默认, 其他值用于 causal/sparse 模式
    ...
)
```

## 返回值

返回元组：`(attention_output, softmax_max, softmax_sum, ...)`

**使用 `[0]` 获取注意力输出张量。**

## 布局说明

| 布局 | 形状 | 使用场景 |
|------|------|---------|
| BNSD | (batch, n_heads, seq_len, head_dim) | 标准多头注意力 |
| BSH | (batch, seq_len, hidden_size) | 融合 QKV |
| SBH | (seq_len, batch, hidden_size) | 序列优先布局 |
| BSND | (batch, seq_len, n_heads, head_dim) | 替代头布局 |
| TND | (total_tokens, n_heads, head_dim) | 变长序列 |

## 常见迁移模式

### 从 `F.scaled_dot_product_attention` 迁移

```python
# 修改前（GPU）
output = F.scaled_dot_product_attention(
    queries, keys, values,
    attn_mask=None,
    dropout_p=self.dropout_p if self.training else 0,
)

# 修改后（NPU）
output = torch_npu.npu_fusion_attention(
    queries, keys, values,
    self.n_heads,
    input_layout="BNSD",
    scale=1.0 / (self.head_dim ** 0.5),
    keep_prob=1.0 - (self.dropout_p if self.training else 0),
)[0]
```

## 与 scaled_dot_product_attention 的关键区别

| 参数 | SDPA（GPU） | npu_fusion_attention（NPU） |
|------|-----------|---------------------------|
| Dropout | `dropout_p`（丢弃概率） | `keep_prob`（保留概率 = 1 - dropout_p） |
| 缩放因子 | 自动计算 | 必须显式提供 |
| 掩码 | `attn_mask` | `atten_mask`（拼写不同） |
| 因果注意力 | `is_causal=True` | `sparse_mode` 参数 |
| 输出 | 直接返回张量 | 返回元组 — 使用 `[0]` |
| 头数 | 不需要 | 必须提供 `head_num` 参数 |
| 布局 | 自动推断 | 必须指定 `input_layout` |

## 因果注意力

对于因果（自回归）注意力：
- 如果原代码使用 `is_causal=True`：使用 `sparse_mode=2` 或提供显式因果掩码
- 如果原代码传入 `None` 掩码且未设置 `is_causal`：非因果注意力，不需要掩码

**根据源代码判断：** 检查原始 `scaled_dot_product_attention` 调用是否使用了 `is_causal=True`。如果是，添加 `sparse_mode=2`。如果传入 `None` 掩码且没有 `is_causal`，则为非因果注意力，不需要掩码。
