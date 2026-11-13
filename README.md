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

**softmax**
```bash
./build/softmax
./build/softmax bench 7
source ~/venv/bin/activate && python softmax_y_check.py
```

## 算子进度

| 算子 | kernel | PyTorch校验 | benchmark |
|---|---|---|---|
| vector_add | ✅ | ✅ | ✅ |
| reduction | ✅ | ✅ | ✅ |
| softmax | ✅| ✅| ✅|
| layernorm | ⬜| ⬜| ⬜|
| GEMM(naive) | ⬜| ⬜| ⬜|
| INT8量化GEMM | ⬜| ⬜| ⬜|

## benchmark 总表

| 算子 | 配置 | 耗时 | 有效带宽 | 对比基准 | 相对基准 |
|---|---|---|---|---|---|
| vector_add | N=2^24+1 BLOCK=256 grid=65537 | ~0.63 ms | ~320 GB/s | PyTorch add(a,c,out=b) 0.62 ms / 325 GB/s | ~98% |
| reduction v1 | N=2^24+1 BLOCK=256 grid=1024 | ~0.21 ms | ~320 GB/s | torch.sum ~0.205 ms / ~327 GB/s | ~98% |
| reduction v2 | warp shuffle,其余同上 | ~0.20 ms | ~330 GB/s | 同上 | ~100% |
| softmax k1 | 4096*4096 一行一block 纯读 | ~0.21 ms | ~320 GB/s | torch.sum 0.205 ms / 327 GB/s | ~98% |
| softmax k2 | 读1写1,8 B/元素 | ~0.50 ms | ~270 GB/s | torch.neg 0.432 ms / 311 GB/s | ~87% |
| softmax k1+k2 | 读2写1,12 B/元素 | ~0.72 ms | ~280 GB/s | torch.softmax 0.617 ms | 慢 16% |
>所有行测于同一会话（2026-11-09）,插电,后台GPU 3D进程已经挂起
>本机跨会话波动可达±10%,绝对值只在同会话内可比

## 算子1 vector_add
### 正确性     

max abs diff = 0.0(逐位相等)· allclose = True

### 性能       

per-call 0.63 ms / 320 GB/s

### 瓶颈分析   

算术强度 1 flop / 12 bytes ≈ 0.083 → 纯带宽受限，可达峰值 98%

### 越界检测   

哨兵法：三个变异全部命中(1020/1024)

### 工具边界

本机 WSL2 不透传 nvperf,ncu 不可用 → 改用"实测可达峰值占比"

## 算子2 reduction

求 N 个 float 之和,N = 2^24+1 = 16777217。

**测量协议**:REPEAT=1000(窗口 ~200 ms),预热后取交替实测均值

| 版本 | 实现方式 | per-call | 有效带宽 | 相对误差 |
|---|---|---|---|---|
| v0 | 1 block × 1 thread 顺序累加 | ~0.43 s | ~0.15 GB/s | 4.17e-2 |
| v1 | 1024 block × 256 thread + 块内 shared 树形归约 | ~0.21 ms | ~320 GB/s | 5.960e-8 |
| v2 | warp shuffle 归约(上同) | ~0.20 ms | ~330 GB/s | 5.960e-8 |
| 标杆 | torch.sum(CUB DeviceReduce) | ~0.205 ms | ~327 GB/s | 5.960e-8 |

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

**分母用 torch.sum 的 ~327 GB/s,不能用 vector_add 的 325 GB/s。**
后者是"读 2 写 1"(12 B/元素)测的,reduction 是**纯读**(4 B/元素)。
写有写回周转开销,纯读流的上限一般更高;本次两侧代理(325 / 327)已经接近,但仍属不同访存口径 —— **不可混用。**

k1 读 67,108,868 B ← 全部 DRAM 流量
k2 读 4,096 B ← 占比 0.006%

~200 µs 里几乎 100% 是 k1 访存 → **归约算法(shared memory / warp shuffle)不是瓶颈。**实测证实: **v1 与 v2 交替实测均值差 0.16%。**

**不再优化** 2026-11-09 同会话实测:v1/v2 ~0.20 ms,torch.sum 0.205 ms,差距<2%,在本机噪声内--没有可追目标

**测量方法的两条硬约束**

**顺序对调对同一个二进制也必需**:同一份 `./build/reduction`,排在第二位的那次慢1.4~4.0 µs;python 侧没有(它要 import torch 3~5 s,启动即稳态)。
**跨会话绝对值一律不可比**:同一份代码跨会话漂 ~10%(`torch.sum` 也一样漂),REPEAT加长只压得住会话内抖动。

### 越界检测

reduction **没有** vector_add 那种尾部越界线程 —— grid-stride 的 `i < n` 天然兜底,所以 vector_add 那套哨兵法在这里不适用。共享内存 `sh[tid]` 的边界由"数组大小 = blockDim"保证。

## 算子3 softmax

4096*4096,对每一行做softmax,输出相同形状

一行一个block,块内256个线程负责整行,两个kernel:k1 求该行的max写入m[row],k2 用m[row]算exp(x-m),块内求和得到S,再除以S写回y

k1,k2 在同一个stream上顺序执行,k2依赖k1的结果,同stream保证顺序,中间不需要cudaDeviceSynchronize,归约全部放在块内,没有跨block汇总

两段式把x读了两遍,这也是后面"融合"的动机来源。(未做)

**测量协议**:插电,WARM=100,k1 REPEAT=1000(~210ms),k2 REPEAT=500(~250ms),k1+k2 REPEAT=300(~215ms);每段两轮;总负载 232/297/286ms。本机连续负载超过0.6s会撞SW Power Cap,带宽明显下降,数字不可用

**测量前挂起后台GPU 3D进程**(本机为动态壁纸):不挂起时k1从325GB/s掉到200GB/s,PyTorch同步下降--与kernel无关

### 正确性

k1的判据是`max abs diff == 0.0`:4096行逐位相等。max是**选择**,不是运算 -- 无论按照什么顺序比较,分几层比较选出来的都是同一个元素,不存在舍入。这一点和vector_add相同,和reduction相反。

k2的判据是`torch.allclose(y,torch.softmax(x,dim=1),rtol=1e-5,atol=1e-6)`:max abs diff ~1.1e-8(约3ulp),每行和与1的偏差~2.4e-7,无NaN。exp,除法和求和都是**运算**,每一步都有舍入，且S的累加顺序和PyTorch不同,结果不可能逐位相同。用`== 0.0`会把正确实现判断成bug。

第0行全部小于-104,专门用来验证"减max":若k2漏掉`-mv`,该行`exp(-120)`下溢为0,S也为0 -> `0/0 = NaN`。

### 性能

| 段 | 访存 | per-call | 有效带宽 | 同会话代理 |
|---|---|---|---|---|
| k1 | 纯读,4 B/元素 | ~0.21 ms | 320 GB/s | torch.sum 0.205 ms / 327 GB/s |
| k2 | 读1写1,8 B/元素 | ~0.50 ms | 270 GB/s | torch.neg 0.432 ms / 311 GB/s |
| k2 去exp | 仅归因,不提交 | ~0.45 ms | ~300 GB/s | 同上 |
| k1 + k2 | 读2写1,12 B/元素 | ~0.72 ms | ~280 GB/s | torch.softmax 0.617 ms|

### 瓶颈分析

**k1到纯读上限** 0.21 ms / 320 GB/s，同会话`torch.sum` 327 GB/s,约98%。k1只读一遍x,没有可以节省的访存,也没有算术负担--无优化空间

**k2的差额全在`expf`** 0.50 ms / 271 GB/s,`torch.neg`为0.432 ms / 311 GB/s(87%)
把`expf`换成普通减法后0.45 ms / 299 GB/s,达到代理的96%,剩余差距在噪声内 -> **访存路径已经到顶,成本来自expf**
k2是读1写1和k1的纯读是不同口径,分母不能混用

**两段式比PyTorch慢16%** 0.715 ms vs `torch.softmax` 0.617 ms
根本原因是x被读了两遍:12 B/元素 若融合为单个kernel -- 读x一次存入shared,在shared上求max,exp和归一化,再写回--访存降低到8 B/元素,**理论预期 ~0.5 ms量级，未做，未验证**