import torch

N = (1 << 24) + 1
REPEAT = 1000

x = torch.arange(N,dtype=torch.float,device="cuda")

for _ in range(100):
	s = torch.sum(x)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(REPEAT):
	s = torch.sum(x)
stop.record()
torch.cuda.synchronize()

ms = start.elapsed_time(stop) / REPEAT
bytes_ = N * 4
print(f"torch.sum per-call = {ms:.4f} ms bandwidth = {bytes_/1e9/(ms*1e-3):.1f}GB/s")

x64 = torch.arange(N,dtype=torch.float64)
ref = x64.sum()
rel = abs(s.double() - ref) / abs(ref)
print(f"torch.sum rel err = {rel.item():.3e}")
