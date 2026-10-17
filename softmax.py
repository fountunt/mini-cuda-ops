import torch

R,C = 4096,4096
x = torch.from_file("x.bin",dtype=torch.float32,size=R*C).reshape(R,C)

print("shape:",tuple(x.shape))
print("全局范围:",x.min().item(),x.max().item())
print("第0行: min=%.3f,max=%.3f" % (x[0].min().item(),x[0].max().item()))
print("第0行全<-104:",bool((x[0] < -104).all()))
print("每行极差最小:",(x.max(1).values - x.min(1).values).min().item())
print("第一行极差:",(x[1].max() - x[1].min()).item())
