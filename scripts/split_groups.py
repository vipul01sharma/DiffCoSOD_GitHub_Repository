import json,argparse,random

def main(a):
 rng=random.Random(a.seed); d=json.load(open(a.input)); ids=list(d); rng.shuffle(ids); k=max(1,int(len(ids)*a.val_fraction)); val={i:d[i] for i in ids[:k]}; train={i:d[i] for i in ids[k:]}; json.dump(train,open(a.train,'w'),indent=2); json.dump(val,open(a.val,'w'),indent=2); print('train groups:',len(train),'val groups:',len(val))
if __name__=='__main__': ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--train',required=True); ap.add_argument('--val',required=True); ap.add_argument('--val-fraction',type=float,default=.1); ap.add_argument('--seed',type=int,default=42); main(ap.parse_args())
