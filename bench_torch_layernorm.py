import torch
import torch.nn.functional as F

R, C = 4096, 4096
WARM = 100
REPEAT = 500

x = torch.from_file("ln_x.bin", dtype=torch.float32, size=R*C).reshape(R, C).cuda()
g = torch.from_file("ln_g.bin", dtype=torch.float32, size=C).cuda()
b = torch.from_file("ln_b.bin", dtype=torch.float32, size=C).cuda()

for _ in range(WARM):
    y = F.layer_norm(x, (C,), g, b)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(REPEAT):
    y = F.layer_norm(x, (C,), g, b)
stop.record()
torch.cuda.synchronize()

ms = start.elapsed_time(stop) / REPEAT
bytes_moved = 2 * R * C * 4
print("per-call = %.4f ms" % ms)
print("bw = %.1f GB/s" % (bytes_moved / 1e9 / (ms * 1e-3)))
print("window = %.1f ms" % (ms * REPEAT))
print("total = %.1f ms" % (ms * (WARM + REPEAT)))