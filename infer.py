import os,argparse
import numpy as np, torch
from PIL import Image
from torch.utils.data import DataLoader
from diffcosod.utils import load_yaml,seed_everything
from diffcosod.data import GroupDataset
from diffcosod.models import DiffCoSOD

def main(a):
    cfg=load_yaml(a.config); seed_everything(cfg['seed']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=DiffCoSOD(cfg).to(device)
    ck=torch.load(a.checkpoint,map_location=device); model.load_state_dict(ck['model']); model.eval(); os.makedirs(a.output_dir,exist_ok=True)
    ds=GroupDataset(a.manifest,cfg['image_size'],cfg['group_size'],False,cfg['seed']); dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=cfg['num_workers'])
    with torch.no_grad():
      for batch in dl:
        imgs=batch['images'].to(device); pred,tr,y0=model.predict(imgs,a.sampling_steps,a.save_trajectory)
        p=pred[0].cpu().numpy()[:,0]
        for name,arr in zip(batch['names'][0],p): Image.fromarray(np.uint8(np.clip(arr,0,1)*255)).save(os.path.join(a.output_dir,os.path.splitext(name)[0]+'.png'))
        if a.save_trajectory:
            np.savez_compressed(os.path.join(a.output_dir,batch['group_id'][0]+'_trajectory.npz'),**{f't_{t}':x.numpy() for t,x in tr})

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--sampling-steps',type=int,default=1000); ap.add_argument('--save-trajectory',action='store_true'); main(ap.parse_args())
