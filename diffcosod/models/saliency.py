import torch.nn as nn
import torch.nn.functional as F

class SFG(nn.Module):
    """Compact SFG-style cross-branch fusion used for the adopted baseline reimplementation."""
    def __init__(self, cnn_channels=(64,256,512,1024,2048), vit_dim=384, out=64):
        super().__init__(); self.cproj=nn.ModuleList([nn.Conv2d(c,out,1) for c in cnn_channels]); self.vproj=nn.Conv2d(vit_dim,out,1); self.fuse=nn.Conv2d(out*2,out,3,padding=1)
    def forward(self, rs, vs):
        outs=[]
        for r,v,cp in zip(rs,vs,self.cproj):
            vv=self.vproj(v); vv=F.interpolate(vv,size=r.shape[-2:],mode='bilinear',align_corners=False); outs.append(self.fuse(nn.functional.gelu(cp(r))+vv))
        return outs

class SaliencyHead(nn.Module):
    def __init__(self,c=64): self_dummy=0; super().__init__(); self.blocks=nn.ModuleList([nn.Conv2d(c,c,3,padding=1) for _ in range(4)]); self.out=nn.Conv2d(c,1,1)
    def forward(self, feats):
        z=feats[-1]
        for f in reversed(feats[:-1]): z=F.interpolate(z,size=f.shape[-2:],mode='bilinear',align_corners=False); z=z+f
        z=F.interpolate(z,size=(352,352),mode='bilinear',align_corners=False); return self.out(z)
