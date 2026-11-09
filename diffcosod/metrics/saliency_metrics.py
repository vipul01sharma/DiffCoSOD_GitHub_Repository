import numpy as np

def mae(p,g): return float(np.abs(p.astype(np.float32)-g.astype(np.float32)).mean())

def fmax(p,g,beta2=0.3):
    best=0.0
    for t in np.linspace(0,1,256):
        q=(p>=t); tp=(q & (g>0.5)).sum(); fp=(q & (g<=0.5)).sum(); fn=((~q) & (g>0.5)).sum(); prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f=(1+beta2)*prec*rec/(beta2*prec+rec+1e-8); best=max(best,float(f))
    return best

def s_measure(p,g,alpha=0.5):
    # Lightweight structural measure approximation; use official benchmark code for publication tables.
    y=p.astype(np.float32); x=g.astype(np.float32); y=(y-y.min())/(y.max()-y.min()+1e-8)
    obj=(2*y.mean()*x.mean()+1e-8)/(y.mean()**2+x.mean()**2+1e-8)
    inter=(y*x).mean(); union=(y+x-y*x).mean(); reg=(inter+1e-8)/(union+1e-8)
    return float(alpha*obj+(1-alpha)*reg)

def e_measure(p,g):
    y=p.astype(np.float32); x=g.astype(np.float32); y=(y-y.mean())/(y.std()+1e-8); x=(x-x.mean())/(x.std()+1e-8); return float(((2*y*x)/(y*y+x*x+1e-8)).mean()*0.5+0.5)
