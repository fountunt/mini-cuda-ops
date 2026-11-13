import numpy as np
import torch
import torch.nn.functional as F

N = 4096

def load(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(shape)

x = load("ln_x.bin", (N, N))
g = load("ln_g.bin", (N,))
b = load("ln_b.bin", (N,))
y = load("ln_y.bin", (N, N))

# 真值：float64 参照（数学定义）
ref64 = F.layer_norm(x.double(), (N,), weight=g.double(), bias=b.double())
# 互操作基准：torch 同精度 float32 实现
ref32 = F.layer_norm(x, (N,), weight=g, bias=b)

e_mine  = (y.double() - ref64).abs()
e_torch = (ref32.double() - ref64).abs()

print("[主判据] vs float64 参照")
print("  allclose(atol=1e-6):", torch.allclose(y.double(), ref64, atol=1e-6, rtol=0))
print("  max abs diff: %.3e" % e_mine.max().item())
print("  第0行: %.3e | 其余行: %.3e" % (e_mine[0].max().item(), e_mine[1:].max().item()))

print("[参照] torch float32 自身 vs float64")
print("  max abs diff: %.3e | 第0行: %.3e" % (e_torch.max().item(), e_torch[0].max().item()))

print("[互操作] vs torch float32")
print("  allclose(atol=1e-4, rtol=1e-4):", torch.allclose(y, ref32, atol=1e-4,rtol=1e-4))
print("  max abs diff: %.3e" % (y - ref32).abs().max().item())

# 独立自检：去掉仿射后每行 mean≈0、std≈1
z = (y - b) / g
print("row mean max |.|: %.3e" % z.mean(dim=1).abs().max().item())
print("row std min/max: %.7f %.7f" % (z.std(dim=1, unbiased=False).min().item(),z.std(dim=1, unbiased=False).max().item()))