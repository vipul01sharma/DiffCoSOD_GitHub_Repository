import torch, yaml
from diffcosod.models.diffusion import ConditionalUNet

def test_unet_shape():
    m=ConditionalUNet(8,16)
    x=torch.randn(2,1,32,32); y=torch.randn_like(x); g=torch.randn(2,16,4,4); t=torch.tensor([1,2])
    z=m(x,t,y,g); assert z.shape==x.shape
