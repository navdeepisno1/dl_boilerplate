import warnings
warnings.filterwarnings('ignore')
import torch 
from models import DiT,DiTConfig,DitModel
from utils.count_params import count_parameters
from tqdm import tqdm

config = DiTConfig(use_linear_attn=False)
model = DitModel(config=config)
print(model)
model.save_pretrained("saved_models/dit_model",safe_serialization=False)
print(count_parameters(model=model))

latent = torch.randn((1,4,64,64))
timesteps = torch.randn((1,320))
context = torch.randn((1,77,768))

import time
s_time = time.time()
for _ in tqdm(range(20)):
    latent = model(latent,timesteps,context)
e_time = time.time()
print((e_time - s_time)/20)
