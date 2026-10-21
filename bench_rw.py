import torch

N = 4096 * 4096
REPEAT = 1000
WARMUP = 100

a = torch.randn(N,device="cuda")
b = torch.empty_like(a);

for _ in range(WARMUP):
	torch.neg(a,out = b)
torch.cuda.synchronize();

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(REPEAT):
	torch.neg(a,out=b)
stop.record()
stop.synchronize()

ms = start.elapsed_time(stop) / REPEAT
gb = 2 * N * 4 / 1e9
print("per-call = %.4f ms" % ms)
print("bandwidth = %.3f GB/s" % (gb / (ms * 1e-3)))
