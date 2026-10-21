import torch

R,C = 4096,4096
x = torch.from_file("x.bin",dtype=torch.float32,size=R*C).reshape(R,C)
y = torch.from_file("y.bin",dtype=torch.float32,size=R*C).reshape(R,C)
y_ref = torch.softmax(x,dim=1)

print("shape:",tuple(y.shape))
print("NaN:",bool(torch.isnan(y).any()))
print("max abs diff:",(y - y_ref).abs().max().item())
print("allclose:",torch.allclose(y,y_ref,rtol=1e-5,atol=1e-6))
s = y.sum(dim=1)
print("row sum: min=%.9f max=%.9f max|sum-1|=%.3e" %
	(s.min().item(),s.max().item(),(s-1).abs().max().item()))
print("y[0][0]=%.9g y[0][C-1]=%.9g ratio=%.3f" %
	(y[0,0].item(),y[0,C-1].item(),(y[0,0]/y[0,C-1]).item()))
print("y[1] max=%.6g min=%.6g sum=%.9f" %
	(y[1].max().item(),y[1].min().item(),y[1].sum().item()))
