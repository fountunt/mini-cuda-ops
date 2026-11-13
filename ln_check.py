import numpy as np
import torch
import torch.nn.functional as F

def load(path,shape):
    return torch.from_numpy(np.fromfile(path,dtype=np.float32)).reshape(shape)

x = load("ln_x.bin",(4096,4096))
g = load("ln_g.bin",(4096,))
b = load("ln_b.bin",(4096,))
y = load("ln_y.bin",(4096,4096))

ref = F.layer_norm(x,(4096,),weight=g,bias=b)
diff = (y - ref).abs()

print("allclose(atol=1e-5, rtol=1e-5):",torch.allclose(y, ref, atol=1e-5, rtol=1e-5))
print("max abs diff:", diff.max().item())
print("mean abs diff:", diff.mean().item())

# 独立自检:去掉仿射后每行应 mean≈0、std≈1(eps 使它略小于 1)
z = (y - b) / g
print("row mean max |.|:", z.mean(dim=1).abs().max().item())
print("row std min/max:", z.std(dim=1, unbiased=False).min().item(),z.std(dim=1, unbiased=False).max().item())
print("row0 mean/std:", z[0].mean().item(), z[0].std(unbiased=False).item())