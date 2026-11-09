import os, argparse, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffcosod.utils import load_yaml, seed_everything, count_parameters
from diffcosod.data import GroupDataset
from diffcosod.models import DiffCoSOD
from diffcosod.losses import total_loss

def move_batch(b,device):
    return {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}

def run(args):
    cfg=load_yaml(args.config); seed_everything(cfg['seed']); cfg['training']['epochs']=args.epochs or cfg['training']['epochs']; cfg['diffusion']['timesteps']=args.timesteps or cfg['diffusion']['timesteps']
    os.makedirs(args.output_dir,exist_ok=True); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=GroupDataset(args.train_manifest,cfg['image_size'],cfg['group_size'],True,cfg['seed']); dl=DataLoader(ds,batch_size=1,shuffle=True,num_workers=cfg['num_workers'])
    model=DiffCoSOD(cfg).to(device); print('trainable parameters:',count_parameters(model)/1e6,'M')
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay']); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['training']['epochs']); scaler=torch.cuda.amp.GradScaler(enabled=cfg['training']['amp'] and device.type=='cuda')
    best=1e9; global_step=0
    for epoch in range(cfg['training']['epochs']):
        model.train(); pbar=tqdm(dl,desc=f'Epoch {epoch+1}/{cfg["training"]["epochs"]}')
        for batch in pbar:
            if args.max_steps and global_step>=args.max_steps: break
            if 'masks' not in batch: continue
            batch=move_batch(batch,device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
                out=model(batch['images'],batch['masks'],False)
                out['_abar']=model.diffusion.abar; out['_t']=torch.randint(0,model.diffusion.timesteps,(batch['images'].shape[0]*batch['images'].shape[1],),device=device)
                B,N=batch['images'].shape[:2]; gt=batch['masks'].reshape(B*N,1,cfg['image_size'],cfg['image_size']); yy=out['coarse'].reshape(B*N,1,cfg['image_size'],cfg['image_size']); gg=out['guidance'].reshape(B*N,out['guidance'].shape[2],out['guidance'].shape[3],out['guidance'].shape[4]); out['diffusion_loss'],out['xt'],out['noise_pred']=model.diffusion.loss(gt,out['_t'],yy,gg)
                loss,parts=total_loss(out,batch['masks'],cfg)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip']); scaler.step(opt); scaler.update(); global_step+=1
            pbar.set_postfix(loss=f'{loss.item():.4f}',diff=f'{parts["diffusion"]:.4f}')
        sch.step()
        torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch+1},os.path.join(args.output_dir,'last.pt'))
        if loss.item()<best: best=loss.item(); torch.save({'model':model.state_dict(),'config':cfg,'epoch':epoch+1},os.path.join(args.output_dir,'best.pt'))
        if args.max_steps and global_step>=args.max_steps: break

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--train-manifest',required=True); ap.add_argument('--val-manifest'); ap.add_argument('--output-dir',default='runs/diffcosod'); ap.add_argument('--epochs',type=int); ap.add_argument('--max-steps',type=int); ap.add_argument('--timesteps',type=int); run(ap.parse_args())
