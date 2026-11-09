import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self,d): super().__init__(); self.d=d; self.mlp=nn.Sequential(nn.Linear(d,d*4),nn.SiLU(),nn.Linear(d*4,d))
    def forward(self,t):
        half=self.d//2; freqs=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/(half-1)); z=t.float()[:,None]*freqs[None,:]; return self.mlp(torch.cat([z.sin(),z.cos()],dim=1))

class ResBlock(nn.Module):
    def __init__(self,cin,cout,td):
        super().__init__(); self.c1=nn.Conv2d(cin,cout,3,padding=1); self.c2=nn.Conv2d(cout,cout,3,padding=1); self.emb=nn.Linear(td,cout); self.n1=nn.GroupNorm(8,cout); self.n2=nn.GroupNorm(8,cout); self.skip=nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
    def forward(self,x,t): z=F.silu(self.n1(self.c1(x))); z=z+self.emb(t)[:,:,None,None]; z=F.silu(self.n2(self.c2(z))); return z+self.skip(x)

class ConditionalUNet(nn.Module):
    def __init__(self,base=32,guidance=64):
        super().__init__(); td=base*8; self.time=TimeEmbedding(td); self.gproj=nn.Conv2d(guidance,base,1); self.inp=nn.Conv2d(2,base,3,padding=1)
        self.b1=ResBlock(base,base,td); self.d1=nn.Conv2d(base,base*2,4,2,1); self.b2=ResBlock(base*2,base*2,td); self.d2=nn.Conv2d(base*2,base*4,4,2,1); self.b3=ResBlock(base*4,base*4,td); self.d3=nn.Conv2d(base*4,base*8,4,2,1); self.mid=ResBlock(base*8,base*8,td)
        self.u3=nn.ConvTranspose2d(base*8,base*4,4,2,1); self.r3=ResBlock(base*8,base*4,td); self.u2=nn.ConvTranspose2d(base*4,base*2,4,2,1); self.r2=ResBlock(base*4,base*2,td); self.u1=nn.ConvTranspose2d(base*2,base,4,2,1); self.r1=ResBlock(base*2,base,td); self.out=nn.Conv2d(base,1,1)
    def forward(self,x,t,y0,g):
        te=self.time(t); g=F.interpolate(self.gproj(g),size=x.shape[-2:],mode='bilinear',align_corners=False); z=self.inp(torch.cat([x,y0],1))+g
        a=self.b1(z,te); b=self.b2(self.d1(a),te); c=self.b3(self.d2(b),te); m=self.mid(self.d3(c),te)
        z=self.r3(torch.cat([self.u3(m),c],1),te); z=self.r2(torch.cat([self.u2(z),b],1),te); z=self.r1(torch.cat([self.u1(z),a],1),te); return self.out(z)

class GaussianDiffusion(nn.Module):
    def __init__(self,model,timesteps=1000,beta_start=1e-4,beta_end=0.02):
        super().__init__(); self.model=model; self.timesteps=timesteps
        betas=torch.linspace(beta_start,beta_end,timesteps); alphas=1-betas; abar=torch.cumprod(alphas,0)
        self.register_buffer('betas',betas); self.register_buffer('alphas',alphas); self.register_buffer('abar',abar); self.register_buffer('sqrt_abar',torch.sqrt(abar)); self.register_buffer('sqrt_one_abar',torch.sqrt(1-abar))
    def q_sample(self,x0,t,noise=None):
        noise=torch.randn_like(x0) if noise is None else noise; a=self.sqrt_abar[t][:,None,None,None]; b=self.sqrt_one_abar[t][:,None,None,None]; return a*x0+b*noise,noise
    def loss(self,gt,t,y0,g):
        xt,noise=self.q_sample(gt,t); pred=self.model(xt,t,y0,g); return F.mse_loss(pred,noise),xt,pred
    @torch.no_grad()
    def sample(self,y0,g,steps=None,return_trajectory=False):
        steps=steps or self.timesteps; device=y0.device; x=torch.randn_like(y0); traj=[]
        # Uniformly map requested inference steps to the trained schedule.
        schedule=torch.linspace(self.timesteps-1,0,steps,device=device).long().unique_consecutive()
        for i,tv in enumerate(schedule):
            t=torch.full((x.shape[0],),int(tv.item()),device=device,dtype=torch.long); eps=self.model(x,t,y0,g); a=self.alphas[t][:,None,None,None]; ab=self.abar[t][:,None,None,None]; beta=self.betas[t][:,None,None,None]
            mean=(x-beta/torch.sqrt(1-ab)*eps)/torch.sqrt(a)
            if int(tv.item())>0: x=mean+torch.sqrt(beta)*torch.randn_like(x)
            else: x=mean
            if return_trajectory and (i==0 or i==len(schedule)//4 or i==len(schedule)//2 or i==3*len(schedule)//4 or i==len(schedule)-1): traj.append((int(tv.item()),x.sigmoid().detach().cpu()))
        return (x.sigmoid(),traj) if return_trajectory else x.sigmoid()
