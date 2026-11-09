import torch
import torch.nn as nn
import torch.nn.functional as F

class SCRL(nn.Module):
    """Saliency-guided pairwise correlation representation (clean reimplementation)."""
    def __init__(self,c=64):
        super().__init__(); self.q=nn.Conv2d(c,c,1); self.k=nn.Conv2d(c,c,1); self.v=nn.Conv2d(c,c,1); self.out=nn.Conv2d(c,c,1)
    def forward(self,f,s):
        b,n,c,h,w=f.shape; x=f.reshape(b*n,c,h*w).transpose(1,2); q=self.q(f.reshape(b*n,c,h,w)).flatten(2).transpose(1,2); k=self.k(f.reshape(b*n,c,h,w)).flatten(2)
        sal=F.interpolate(s,size=(h,w),mode='bilinear',align_corners=False).reshape(b*n,1,h*w).transpose(1,2)
        a=torch.softmax((q@k)/(c**0.5)+sal,dim=-1); v=self.v(f.reshape(b*n,c,h,w)).flatten(2).transpose(1,2); z=(a@v).transpose(1,2).reshape(b*n,c,h,w)
        return self.out(z).reshape(b,n,c,h,w)

class CCRL(nn.Module):
    """Group consensus representation: attention-weighted group aggregation."""
    def __init__(self,c=64):
        super().__init__(); self.score=nn.Conv2d(c,1,1); self.proj=nn.Conv2d(c,c,1)
    def forward(self,f):
        b,n,c,h,w=f.shape; scores=self.score(f.reshape(b*n,c,h,w)).reshape(b,n,1,h,w); a=torch.softmax(scores,dim=1); cons=(a*f).sum(dim=1,keepdim=True); cons=self.proj(cons.reshape(b,c,h,w)).unsqueeze(1); return cons.expand(-1,n,-1,-1,-1)

class CorrelationLearning(nn.Module):
    def __init__(self,c=64):
        super().__init__(); self.scrl=SCRL(c); self.ccrl=CCRL(c); self.fuse=nn.Conv2d(c*2,c,3,padding=1); self.decoder=nn.Sequential(nn.Conv2d(c,c,3,padding=1),nn.GELU(),nn.Conv2d(c,1,1))
    def forward(self, feats, sal):
        f=feats[-1]; b,n,c,h,w=f.shape; fs=self.scrl(f,sal); fc=self.ccrl(f); z=torch.cat([fs,fc],dim=2).reshape(b*n,2*c,h,w); z=self.fuse(z); y=self.decoder(z); y=F.interpolate(y,size=sal.shape[-2:],mode='bilinear',align_corners=False).reshape(b,n,1,sal.shape[-2],sal.shape[-1]); return fs,fc,y
