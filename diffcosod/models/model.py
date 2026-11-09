import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import DualEncoder
from .saliency import SFG
from .correlation import CorrelationLearning
from .gadg import GADG
from .diffusion import ConditionalUNet, GaussianDiffusion

class DiffCoSOD(nn.Module):
    def __init__(self,cfg):
        super().__init__(); m=cfg['model']; d=cfg['diffusion']; c=m['feature_dim']
        self.encoder=DualEncoder(m['resnet_pretrained'],m['vit_dim'],m['vit_depth'],m['vit_heads'],m['vit_patch'],cfg['image_size']); self.sfg=SFG((64,256,512,1024,2048),m['vit_dim'],c); self.sal_head=nn.Conv2d(c,1,1); self.corr=CorrelationLearning(c); self.gadg=GADG(c,m['guidance_dim']); self.diffusion=GaussianDiffusion(ConditionalUNet(m['diffusion_base_channels'],m['guidance_dim']),d['timesteps'],d['beta_start'],d['beta_end'])
    def forward_features(self,images):
        B,N,C,H,W=images.shape; x=images.reshape(B*N,C,H,W); rs,vs=self.encoder(x); fs=self.sfg(rs,vs); # saliency prediction from fused multi-scale features
        z=fs[-1]
        for f in reversed(fs[:-1]): z=F.interpolate(z,size=f.shape[-2:],mode='bilinear',align_corners=False)+f
        sal=F.interpolate(self.sal_head(z),size=(H,W),mode='bilinear',align_corners=False).reshape(B,N,1,H,W)
        feat=fs[-1]; h,w=feat.shape[-2:]; feat=feat.reshape(B,N,feat.shape[1],h,w); sal_low=F.interpolate(sal.reshape(B*N,1,H,W),size=(h,w),mode='bilinear',align_corners=False).reshape(B,N,1,h,w)
        fs_corr,fc_corr,y0=self.corr([feat],sal_low)
        g=self.gadg(sal,fs_corr,fc_corr)
        return sal,y0,fs_corr,fc_corr,g
    def forward(self,images,masks=None,training_diffusion=True,t=None):
        sal,y0,fs,fc,g=self.forward_features(images); out={'saliency':sal,'coarse':y0,'fs':fs,'fc':fc,'guidance':g}
        if masks is not None and training_diffusion:
            B,N,_,H,W=masks.shape; gt=masks.reshape(B*N,1,H,W); yy=y0.reshape(B*N,1,H,W); gg=g.reshape(B*N,g.shape[2],g.shape[3],g.shape[4]); t=torch.randint(0,self.diffusion.timesteps,(gt.shape[0],),device=gt.device) if t is None else t
            ld,xt,eps=self.diffusion.loss(gt,t,yy,gg); out.update(diffusion_loss=ld,xt=xt,noise_pred=eps)
        return out
    @torch.no_grad()
    def predict(self,images,steps=None,trajectory=False):
        sal,y0,fs,fc,g=self.forward_features(images); B,N,_,H,W=y0.shape; pred=self.diffusion.sample(y0.reshape(B*N,1,H,W),g.reshape(B*N,g.shape[2],g.shape[3],g.shape[4]),steps,trajectory); 
        if trajectory: p,tr=pred; return p.reshape(B,N,1,H,W),tr,y0
        return pred.reshape(B,N,1,H,W)
