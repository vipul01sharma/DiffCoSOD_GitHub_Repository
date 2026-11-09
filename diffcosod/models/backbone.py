import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class MLP(nn.Module):
    def __init__(self, d, hidden): super().__init__(); self.net=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
    def forward(self,x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d=384, heads=6):
        super().__init__(); self.n1=nn.LayerNorm(d); self.attn=nn.MultiheadAttention(d,heads,batch_first=True); self.n2=nn.LayerNorm(d); self.mlp=MLP(d,d*4)
    def forward(self,x):
        z=self.n1(x); x=x+self.attn(z,z,z,need_weights=False)[0]; return x+self.mlp(self.n2(x))

class CompactViT(nn.Module):
    def __init__(self, dim=384, depth=6, heads=6, patch=16, image_size=352):
        super().__init__(); self.patch=patch; self.dim=dim
        self.proj=nn.Conv2d(3,dim,patch,patch); n=(image_size//patch)**2
        self.pos=nn.Parameter(torch.zeros(1,n,dim)); self.blocks=nn.ModuleList([TransformerBlock(dim,heads) for _ in range(depth)]); self.norm=nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos,std=.02)
    def forward(self,x):
        z=self.proj(x).flatten(2).transpose(1,2)+self.pos[:, :((x.shape[-1]//self.patch)**2)]
        for b in self.blocks: z=b(z)
        z=self.norm(z); h=w=x.shape[-1]//self.patch
        return z.transpose(1,2).reshape(x.shape[0],self.dim,h,w)

class DualEncoder(nn.Module):
    """Manuscript-aligned dual encoder: ResNet-50 local features + compact 384-d ViT global features."""
    def __init__(self, pretrained=True, vit_dim=384, vit_depth=6, vit_heads=6, patch=16, image_size=352):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        r=resnet50(weights=weights)
        self.stem=nn.Sequential(r.conv1,r.bn1,r.relu,r.maxpool); self.layer1=r.layer1; self.layer2=r.layer2; self.layer3=r.layer3; self.layer4=r.layer4
        self.vit=CompactViT(vit_dim,vit_depth,vit_heads,patch,image_size)
    def forward(self,x):
        z=self.stem(x); r1=self.layer1(z); r2=self.layer2(r1); r3=self.layer3(r2); r4=self.layer4(r3)
        v=self.vit(x)
        # five-level contract: shallow through deep CNN plus a global ViT map
        return [z,r1,r2,r3,r4], [F.interpolate(v,size=t.shape[-2:],mode='bilinear',align_corners=False) for t in [z,r1,r2,r3,r4]]
