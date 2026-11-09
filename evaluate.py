import os,argparse,json,numpy as np
from PIL import Image
from diffcosod.metrics import mae,fmax,s_measure,e_measure

def main(a):
 with open(a.manifest) as f: groups=json.load(f)
 rows=[]
 root=os.path.dirname(os.path.abspath(a.manifest))
 for gid,recs in groups.items():
  for r in recs:
   if 'mask' not in r: continue
   pred=os.path.join(a.pred_dir,os.path.splitext(os.path.basename(r['image']))[0]+'.png');
   if not os.path.exists(pred): continue
   p=np.asarray(Image.open(pred).convert('L'),dtype=np.float32)/255.; g=np.asarray(Image.open(r['mask'] if os.path.isabs(r['mask']) else os.path.join(root,r['mask'])).convert('L').resize((p.shape[1],p.shape[0])),dtype=np.float32)/255.; rows.append([mae(p,g),fmax(p,g),s_measure(p,g),e_measure(p,g)])
 if not rows: raise RuntimeError('No matched prediction/mask pairs found.')
 x=np.asarray(rows); print(json.dumps({'MAE':float(x[:,0].mean()),'Fmax':float(x[:,1].mean()),'S':float(x[:,2].mean()),'E':float(x[:,3].mean()),'images':len(rows)},indent=2))
if __name__=='__main__': ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--pred-dir',required=True); main(ap.parse_args())
