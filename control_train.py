"""Deterministic refinement control for the reviewer-requested diffusion-vs-refinement test."""
import argparse, os, torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from diffcosod.utils import load_yaml, seed_everything
from diffcosod.data import GroupDataset
from diffcosod.models.model import DiffCoSOD
from torch.utils.data import DataLoader

class DeterministicRefiner(nn.Module):
    def __init__(self,c=64):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(2+c,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,3,padding=1))
    def forward(self,y,y0,g): return y + self.net(torch.cat([y,y0,g],1))

def main(a):
 cfg=load_yaml(a.config); seed_everything(cfg['seed']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); base=DiffCoSOD(cfg).to(device)
 # Freeze the shared adopted saliency/correlation trunk for a clean refinement control.
 for p in base.parameters(): p.requires_grad=False
 ref=DeterministicRefiner(cfg['model']['guidance_dim']).to(device); opt=torch.optim.AdamW(ref.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay']); ds=GroupDataset(a.train_manifest,cfg['image_size'],cfg['group_size'],True,cfg['seed']); dl=DataLoader(ds,batch_size=1,shuffle=True,num_workers=cfg['num_workers']); os.makedirs(a.output_dir,exist_ok=True)
 for ep in range(a.epochs):
  for b in tqdm(dl,desc=f'control {ep+1}/{a.epochs}'):
   imgs=b['images'].to(device); gt=b['masks'].to(device)
   with torch.no_grad(): sal,y0,fs,fc,g=base.forward_features(imgs)
   B,N=imgs.shape[:2]; y=y0.reshape(B*N,1,cfg['image_size'],cfg['image_size']); gg=g.reshape(B*N,g.shape[2],g.shape[3],g.shape[4]); target=gt.reshape(B*N,1,cfg['image_size'],cfg['image_size']);
   for _ in range(a.refine_steps): y=torch.sigmoid(ref(y,y0.reshape(B*N,1,cfg['image_size'],cfg['image_size']),gg))
   loss=F.binary_cross_entropy(y,target); opt.zero_grad(); loss.backward(); opt.step()
  torch.save({'model':ref.state_dict(),'config':cfg,'epochs':a.epochs},os.path.join(a.output_dir,'control.pt'))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--train-manifest',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--refine-steps',type=int,default=10); main(ap.parse_args())
