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

**vector_add**
```bash
source ~/venv/bin/activate
python verify.py
```

**reduction**
```bash
./build/reduction 1
source ~/venv/bin/activate && python verify_reduction.py
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
| reduction v1 | N=2^24+1 BLOCK=256 grid=1024 | ~0.23 ms | ~290 GB/s | torch.sum ~0.225 ms / ~299 GB/s | ~%96 |
| reduction v2 | warp shuffle,其余同上 | ~0.23 ms | ~290 GB/s | 同上 | ~96% |
## 算子1 vector_add
### 正确性     

max abs diff = 0.0(逐位相等)· allclose = True

### 性能       

per-call 0.6777 ms · 297.1 GB/s

### 瓶颈分析   

算术强度 1 flop / 12 bytes ≈ 0.083 → 纯带宽受限，可达峰值 91~97%

### 越界检测   

哨兵法：三个变异全部命中(1020/1024)

### 工具边界

本机 WSL2 不透传 nvperf,ncu 不可用 → 改用"实测可达峰值占比"

## 算子2 reduction

求 N 个 float 之和,N = 2^24+1 = 16777217。

**测量协议**:REPEAT=1000(窗口 226 ms),预热后取交替实测均值

| 版本 | 实现方式 | per-call | 有效带宽 | 相对误差 |
|---|---|---|---|---|
| v0 | 1 block × 1 thread 顺序累加 | ~0.43 s | ~0.15 GB/s | 4.17e-2 |
| v1 | 1024 block × 256 thread + 块内 shared 树形归约 | ~0.226 ms | ~297 GB/s | 5.960e-8 |
| v2 | warp shuffle 归约(上同) | ~0.226 ms | ~297 GB/s | 5.960e-8 |
| 标杆 | torch.sum(CUB DeviceReduce) | ~0.225 ms | ~299 GB/s | 5.960e-8 |

`./build/reduction` 选择版本与重复次数,所有版本共用同一份 host脚手架与输入。

### 正确性

判据是**相对误差**(参考 = float64 累加),**不是** vector_add 用的 `max abs diff == 0.0`。
原因:浮点加法不满足结合律,`((a0+a1)+a2)` 与 `(a0+a1)+(a2+a3)` 结果必然不同。

**v1、v2 与 torch.sum 的输出逐位相同**(`140737505132544.0`)。

### 数值精度:5.960e-8 是硬下限,不是调优结果

精确和 = 2^23 × 16777217 = 140737496743936
最近的两个 float32 = 2^47 与 2^47 + 2^24
精确和离这两个的距离 = 都是 2^23 ← 恰好落在正中间
→ **任何 float32 实现都满足 rel err ≥ 2^-24 = 5.96e-8**,不存在更好的答案。这个下限是**该输入(N=2^24+1,x[i]=i)特有的**,换个 N 就变。

### 测量方法

**窗口长度** 本机 GPU 时钟动态调节,窗口太短会把时钟爬升段算进去。

| REPEAT | 窗口 | 实测极差 |
|---|---|---|
| 10 | 2.6 ms | ~25% |
| 100 | 22.6 ms | 12 ~ 15% |
| **1000** | **226 ms** | **1.6%** |

选 REPEAT 的判据不是"跑得快",是**窗口长到启动瞬态只占一小部分**。本机取窗口 ≥ 200ms。

**交替 + 多轮** 两个版本必须"同一 shell 里紧挨着跑",共享同一份漂移,相减时抵消。只看单次采样会得到 25% 的假差异。

**顺序对调** 跨语言比较(python vs C++)时,**进程的排列位置本身带来约 5%的系统偏差** ——紧跟 python 进程之后测的 v1 是 0.238~0.241,紧跟另一个 reduction 进程之后是0.226~0.227。奇数轮 A→B、偶数轮 B→A,让两边各有相同次数在前在后,偏差在总均值里抵消。

**跨会话漂移** 同一份代码在不同会话实测可差 **~6.5%**(温度 / 电源 / 后台负载)REPEAT 加长只解决"会话内"的抖动,解决不了"会话之间"。**绝对值只在同一会话内有意义。**

### 瓶颈分析

算术强度 = 1 flop / 4 bytes = 0.25 flop/byte   → 纯带宽受限

**分母用 torch.sum 的 ~299 GB/s,不能用 vector_add 的 306.7 GB/s。**
后者是"读 2 写 1"(12 B/元素)测的,reduction 是**纯读**(4 B/元素)。
写有写回周转开销,纯读流的上限更高 —— **口径不同,不可混用。**

k1 读 67,108,868 B ← 全部 DRAM 流量
k2 读 4,096 B ← 占比 0.006%

226 µs 里几乎 100% 是 k1 访存 → **归约算法(shared memory / warp shuffle)不是瓶颈。**实测证实: **v1 与 v2 交替实测均值差 0.16%。**

**与 torch.sum 的对比(两轮交替实测):**

| 会话 | 顺序 | torch.sum 均值 | v1 均值 | 差 |
|---|---|---|---|---|
| 第 1 次 | 固定单序 | 0.2229 ms | 0.2351 ms | 5.5% |
| 第 2 次 | **奇偶对调** | 0.2251 ms | 0.2329 ms | **3.5%** |

**但过程中量到一个同等量级的测量偏差**:
跨语言比较(python vs `./build/reduction`)时,**进程的排列位置会造成约 5% 的系统偏差**——紧跟在 python 进程之后测的 v1 是 0.238~0.241,紧跟在另一个 reduction 进程之后测的是0.226~0.227。

**差距归因(2026-10-11,`ver 3` 探针实测)**

`ver 3` 只跑 k1、不跑 k2,把差距拆成两笔:

| 成分 | 实测 | 占 8.6 µs |
|---|---|---|
| **k2 的固定开销**(launch 延迟 + 单 block 归约) | **4.65 µs(1.8%)** | 54% |
| **k1 访存方式 vs CUB**(标量 vs `float4`) | 1.77 µs(0.7%) | 21% |
| 未归属(会话内漂移) | ~2.2 µs | 25% |

**结论:不再优化。** 单 kernel(消掉 k2 的 launch)净收益 ≈1%,`float4` ≤0.7% ——两者都低于本机分辨力:**同一个二进制只因排在第二位就慢 1.4~4.0 µs(约 1%)。**v1/v2 已到外部标杆的 **~96%**,到此为止。

**测量方法的两条硬约束**

**顺序对调对同一个二进制也必需**:同一份 `./build/reduction`,排在第二位的那次慢1.4~4.0 µs;python 侧没有(它要 import torch 3~5 s,启动即稳态)。
**跨会话绝对值一律不可比**:同一份代码跨会话漂 ~10%(`torch.sum` 也一样漂),REPEAT加长只压得住会话内抖动。

### 越界检测

reduction **没有** vector_add 那种尾部越界线程 —— grid-stride 的 `i < n` 天然兜底,所以 vector_add 那套哨兵法在这里不适用。共享内存 `sh[tid]` 的边界由"数组大小 = blockDim"保证。
