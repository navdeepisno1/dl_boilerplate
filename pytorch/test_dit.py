import warnings
warnings.filterwarnings('ignore')
import torch 
from models import DiT,DiTConfig
from utils.count_params import count_parameters

config = DiTConfig(use_linear_attn=False)
model = DiT(config=config)
# torch.save(model.state_dict(),"dit.pth")
# print(count_parameters(model=model))

latent = torch.randn((1,4,64,64))
timesteps = torch.randn((1,320))
context = torch.randn((1,77,768))

import time
for _ in range(20):
    s_time = time.time()
    latent = model(latent,timesteps,context)
    e_time = time.time()
    print(latent.shape, (e_time - s_time))
