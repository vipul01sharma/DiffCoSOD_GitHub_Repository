"""Create a group manifest from a CSV with columns: group,image,mask."""
import csv,json,argparse,os

def main(a):
 groups={}
 with open(a.csv,newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f): groups.setdefault(r['group'],[]).append({'image':r['image'],'mask':r.get('mask','')})
 with open(a.output,'w',encoding='utf-8') as f: json.dump(groups,f,indent=2)
 print('groups:',len(groups),'records:',sum(map(len,groups.values())))
if __name__=='__main__': ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--output',required=True); main(ap.parse_args())
