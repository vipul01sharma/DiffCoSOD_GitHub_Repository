import argparse,json,time,torch,os
from torch.utils.data import DataLoader
from diffcosod.utils import load_yaml,seed_everything
from diffcosod.data import GroupDataset
from diffcosod.models import DiffCoSOD

def main(a):
 cfg=load_yaml(a.config); seed_everything(cfg['seed']); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=DiffCoSOD(cfg).to(device); ck=torch.load(a.checkpoint,map_location=device); model.load_state_dict(ck['model']); model.eval(); ds=GroupDataset(a.manifest,cfg['image_size'],cfg['group_size'],False,cfg['seed']); b=next(iter(DataLoader(ds,batch_size=1,num_workers=0)))['images'].to(device)
 results=[]
 for steps in a.sampling_steps:
  for _ in range(a.warmup):
   with torch.no_grad(): model.predict(b,steps)
  if device.type=='cuda': torch.cuda.synchronize(); start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True); start.record();
  else: start_time=time.perf_counter()
  with torch.no_grad(): model.predict(b,steps)
  if device.type=='cuda': end.record(); torch.cuda.synchronize(); sec=start.elapsed_time(end)/1000
  else: sec=time.perf_counter()-start_time
  mem=torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0
  results.append({'timesteps':steps,'seconds':sec,'fps':1/sec,'memory_gb':mem})
 os.makedirs(a.output_dir,exist_ok=True); json.dump(results,open(os.path.join(a.output_dir,'runtime.json'),'w'),indent=2); print(json.dumps(results,indent=2))
if __name__=='__main__': ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--output-dir',default='runs/runtime'); ap.add_argument('--sampling-steps',type=int,nargs='+',default=[250,500,750,1000]); ap.add_argument('--warmup',type=int,default=2); main(ap.parse_args())
