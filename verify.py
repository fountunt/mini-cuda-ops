import torch

N = (1 << 24) + 1

raw = bytearray(open("c.bin","rb").read())
c_cuda = torch.frombuffer(raw,dtype=torch.float32)
assert c_cuda.numel() == N,f"元素个数对不上:{c_cuda.numel()} vs {N}"

i = torch.arange(N,dtype=torch.float32)
c_ref = i * 0.5 + i * 0.25

assert c_ref.dtype == c_cuda.dtype,(
	f"dtype 不一致:参考是 {c_ref.dtype},kernel 是 {c_cuda.dtype}"
)

print("c_cuda[0:5]  =",c_cuda[:5].tolist())
print("c_ref [0:5]  =",c_ref[:5].tolist())
print("c_cuda[-5:]  =",c_cuda[-5:].tolist())
print("c_ref [-5:]  =",c_ref[-5:].tolist())
print("max abs diff =",(c_cuda - c_ref).abs().max().item())
print("allclose     =",torch.allclose(c_cuda,c_ref,atol=1e-6))
