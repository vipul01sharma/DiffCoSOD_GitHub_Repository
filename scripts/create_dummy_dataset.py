"""Create a tiny synthetic grouped dataset for smoke tests only."""
import argparse,os,json
from PIL import Image,ImageDraw

def main(a):
 os.makedirs(a.output,exist_ok=True); groups={}
 for g in range(a.groups):
  items=[]
  for i in range(a.group_size):
   name=f'g{g:03d}_i{i:02d}'; im=Image.new('RGB',(128,128),'gray'); m=Image.new('L',(128,128),0); d=ImageDraw.Draw(im); dm=ImageDraw.Draw(m); x=15+i*3; y=25+g*2; d.ellipse((x,y,x+70,y+55),fill='white'); dm.ellipse((x,y,x+70,y+55),fill=255); im.save(os.path.join(a.output,name+'.jpg')); m.save(os.path.join(a.output,name+'.png')); items.append({'image':name+'.jpg','mask':name+'.png'})
  groups[f'group_{g:03d}']=items
 json.dump(groups,open(os.path.join(a.output,'manifest.json'),'w'),indent=2)
 print(os.path.join(a.output,'manifest.json'))
if __name__=='__main__': ap=argparse.ArgumentParser(); ap.add_argument('--output',default='examples/data'); ap.add_argument('--groups',type=int,default=2); ap.add_argument('--group-size',type=int,default=5); main(ap.parse_args())
