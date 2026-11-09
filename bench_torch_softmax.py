import torch

R,C = 4096,4096
WARM = 100
REPEAT = 500

x = torch.from_file("x.bin", dtype=torch.float32, size=R*C).reshape(R, C).cuda()

for _ in range(WARM):
	y = torch.softmax(x, dim=1)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(REPEAT):
	y = torch.softmax(x, dim=1)
stop.record()
torch.cuda.synchronize()

ms = start.elapsed_time(stop) / REPEAT
print("per-call = %.4f ms" % ms)
print("window = %.1f ms" % (ms * REPEAT))
print("total = %.1f ms" % (ms * (WARM + REPEAT)))
