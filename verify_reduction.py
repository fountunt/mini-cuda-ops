import torch

N = (1 << 24) + 1;

raw = bytearray(open("sum.bin","rb").read())
assert len(raw) == 4,f"sum.bin 应该4字节，实际{len(raw)}"
sum_cuda = torch.frombuffer(raw,dtype=torch.float32)[0]
x32 = torch.arange(N,dtype=torch.float32)
x64 = x32.to(torch.float64)
sum_ref = x64.sum()

assert sum_ref.dtype == torch.float64

rel = abs(sum_cuda.double() - sum_ref) / abs(sum_ref)

print(f"sum_cuda = {sum_cuda.item():.9g} (float32)")
print(f"sum_ref = {sum_ref.item():.17g} (float64,精确值)")
print(f"rel err = {rel.item():.3e}")
print(f"理论下限 = 5.96e-08 <- float32 表示极限，任何实现都低于不了")

