# mini-cuda-ops

用CUDA手写大模型推理的核心算子，每个算子给出 正确性校验+性能数字+瓶颈分析

## 环境

GPU:RTX 5060 Laptop(8GB,Blackwell,**`sm_120`**)
CUDA 12.8
系统:WSL2 + Ubuntu 24.04
PyTorch 2.x

## 编译与运行

依赖:CMake >= 3.20,CUDA 12.8,GCC 13

**在仓库根目录下执行**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vector_add
```

```bash
source ~/venv/bin/activate
python verify.py
```

## 算子进度

| 算子 | kernel | PyTorch校验 | benchmark |
|---|---|---|---|
| vector_add | ✅ | ✅ | ✅ |
| reduction | ✅ | ✅ | ✅ |
| softmax | ⬜| ⬜| ⬜|
| layernorm | ⬜| ⬜| ⬜|
| GEMM(naive) | ⬜| ⬜| ⬜|
| INT8量化GEMM | ⬜| ⬜| ⬜|

## benchmark 总表

| 算子 | 配置 | 耗时 | 有效带宽 | 对比基准 | 相对基准 |
|---|---|---|---|---|---|
| vector_add | N=2^24+1 BLOCK=256 grid=65537 | 0.6777 ms | 297.1 GB/s | PyTorch add(a,c,out=b) 实测可达峰值306.7 GB/s | 96.9% |

## 算子1 vector_add
### 正确性     

max abs diff = 0.0(逐位相等)· allclose = True

### 性能       

per-call 0.6777 ms · 297.1 GB/s

### 瓶颈分析   

算术强度 1 flop / 12 bytes ≈ 0.083 → 纯带宽受限，可达峰值 91~97% → 无优化空间

### 越界检测   

哨兵法：三个变异全部命中(1020/1024)

### 工具边界

本机 WSL2 不透传 nvperf,ncu 不可用 → 改用"实测可达峰值占比"

## 算子2 reduction

求N个float之和，N = 2^24+1

| 版本 | 实现方式 | per-call | 有效带宽 | 相对误差 |
|---|---|---|---|---|
| v0 | 1 线程顺序累加 | 418.7 ms | 0.160 GB/s | 4.17e-2 |
| v1 | 多block + 块内树形归约 | 0.2128ms | 315.4GB/s | 5.960e-8 |
| 标杆 | torch.sum | 0.2167ms | 309.6GB/s | 5.960e-8 |

`./build/reduction 0/1/2` 选择版本，共用一份host脚手架与一份输入

### 正确性

v1与torch.sum的输出逐位相同(`140737505132544.0`)

### 数值精度：5.960e-8是硬下限，不是调优结果

精确和 = 2^23 * 16777217
最近两个 float32 = 2^47 和 2^47 + 2^24
精确和距离都为2^23 -- 恰好落在正中间

### 瓶颈分析

算数强度 = 1flop / 4 bytes = 0.25 flop/byte 纯宽带受限

**分母用 torch.sum 的309.6GB/s 而不是vector_add的306.7GB/s**
后者是"读2写1" reduction是"纯读取"
口径不同 不可混用

v1 达到 315.4 GB/s = torch.sum 的 ~102%
比 torch.sum 快 1.8% ≪ 本机噪声底 ±6%
→ 判定"达到 torch.sum 水平",不声称更快

**已无优化空间,依据是实测的流量构成:**

k1 读 67,108,868 B ← 全部 DRAM 流量
k2 读 4,096 B ← 占比 0.006%

213 µs 里几乎 100% 是 k1 访存,归约的数学量相对每线程 64 次全局读可忽略

### 越界检测

reduction **没有** vector_add 那种尾部越界线程 —— grid-stride 的 `i < n` 天然兜底,
所以 vector_add 那套哨兵法在这里不适用。共享内存 `sh[tid]` 的边界由"数组大小 = blockDim"保证。
