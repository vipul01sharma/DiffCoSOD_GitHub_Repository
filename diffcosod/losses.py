import torch
import torch.nn.functional as F

def iou_loss(logits, target, eps=1e-6):
    p=torch.sigmoid(logits); inter=(p*target).flatten(1).sum(1); union=(p+target-p*target).flatten(1).sum(1); return (1-(inter+eps)/(union+eps)).mean()

def seg_loss(logits,target): return F.binary_cross_entropy_with_logits(logits,target)+iou_loss(logits,target)

def total_loss(out,masks,cfg):
    gt=masks; sal=out['saliency']; coarse=out['coarse']; ld=out['diffusion_loss'];
    ls=seg_loss(sal.reshape(-1,1,*sal.shape[-2:]),gt.reshape(-1,1,*gt.shape[-2:]))
    lc=seg_loss(coarse.reshape(-1,1,*coarse.shape[-2:]),gt.reshape(-1,1,*gt.shape[-2:]))
    abar=out['_abar']; t=out['_t']; xt=out['xt']; eps=out['noise_pred']; a=abar[t][:,None,None,None]; b=torch.sqrt(1-a); x0=(xt-b*eps)/(torch.sqrt(a)+1e-8)
    lr=seg_loss(x0,gt.reshape(-1,1,*gt.shape[-2:]))
    w=cfg['training']; return w['lambda_saliency']*ls+w['lambda_cosaliency']*lc+w['lambda_diffusion']*ld+w['lambda_refinement']*lr, {'saliency':ls.item(),'coarse':lc.item(),'diffusion':ld.item(),'refinement':lr.item()}
