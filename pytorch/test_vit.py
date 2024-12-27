import warnings
warnings.filterwarnings('ignore')
import torch 
from models import ViTModel,VitConfig
from utils.count_params import count_parameters
from tqdm import tqdm

config = VitConfig()
model = ViTModel(config=config)
print(model)
model.save_pretrained("saved_models/vit_model")
print(count_parameters(model=model))

image = torch.randn(1,3,448,448)
import time
s_time = time.time()
# for _ in tqdm(range(20)):
image = model(image)
e_time = time.time()
print((e_time - s_time)/20)
