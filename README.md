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
| reduction | ⬜| ⬜| ⬜|
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

| 版本 | 实现方式 | per-call | 有效带宽 | 相对误差 |
|---|---|---|---|---|
| v0 | 1 线程顺序累加 | 418.7 ms | 0.160 GB/s | 4.17e-2 |
