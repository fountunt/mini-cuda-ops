import torch

R,C = 4096,4096
x = torch.from_file("x.bin",dtype=torch.float,size=R*C).reshape(R,C)
m = torch.from_file("m.bin",dtype=torch.float,size=R)

m_ref = x.max(dim=1).values
d = (m - m_ref).abs()

print("max abs diff:",d.max().item())
print("完全相等行数:",int((d == 0).sum()),"/",R)
print("m[:3] = ",m[:3].tolist())
print("m_ref[:3] = ",m_ref[:3].tolist())
