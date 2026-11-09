import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention2D(nn.Module):
    def __init__(self,c,heads=4):
        super().__init__(); self.attn=nn.MultiheadAttention(c,heads,batch_first=True); self.n=nn.LayerNorm(c)
    def forward(self,q,kv):
        b,c,h,w=q.shape; qq=q.flatten(2).transpose(1,2); kk=kv.flatten(2).transpose(1,2); z=self.attn(self.n(qq),self.n(kk),self.n(kk),need_weights=False)[0]; return (qq+z).transpose(1,2).reshape(b,c,h,w)

class GADG(nn.Module):
    """Group-Aware Diffusion Guidance: align, fuse and cross-attend saliency/correlation features."""
    def __init__(self,c=64,out=64):
        super().__init__(); self.ps=nn.Conv2d(c,out,1); self.pf=nn.Conv2d(c,out,1); self.pc=nn.Conv2d(c,out,1); self.fuse=nn.Sequential(nn.Conv2d(out*3,out,1),nn.GELU(),nn.Conv2d(out,out,3,padding=1)); self.attn=CrossAttention2D(out)
    def forward(self,s,fs,fc):
        # s: B,N,1,H,W; fs/fc: B,N,C,h,w
        b,n,_,H,W=s.shape; target=(H//8,W//8); sm=F.interpolate(s.reshape(b*n,1,H,W),size=target,mode='bilinear',align_corners=False).repeat(1,self.ps.in_channels,1,1)
        fs=fs.reshape(b*n,fs.shape[2],fs.shape[3],fs.shape[4]); fc=fc.reshape(b*n,fc.shape[2],fc.shape[3],fc.shape[4]); fs=F.interpolate(fs,size=target,mode='bilinear',align_corners=False); fc=F.interpolate(fc,size=target,mode='bilinear',align_corners=False)
        g=self.fuse(torch.cat([self.ps(sm),self.pf(fs),self.pc(fc)],dim=1)); g=self.attn(g,g); return g.reshape(b,n,g.shape[1],g.shape[2],g.shape[3])
