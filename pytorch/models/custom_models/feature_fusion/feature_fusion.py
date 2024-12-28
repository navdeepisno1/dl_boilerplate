import torch
import torch.nn as nn
from typing import List

class FeatureFusionBlock(nn.Module):
    def __init__(self,in_features_list:List[int],out_features:int):
        super(FeatureFusionBlock, self).__init__()
        self.linear_mean_layers = nn.ModuleList(
            [
                nn.Linear(in_features=i,out_features=out_features) for i in in_features_list
            ]
        )
        self.linear_sigma_layers = nn.ModuleList(
            [
                nn.Linear(in_features=i,out_features=out_features) for i in in_features_list
            ]
        )

    def forward(self, x:List[torch.Tensor]):
        assert len(x)==len(self.linear_mean_layers), "Inputs must be equal to the layers provided"

        mu_s = 0
        for i,layer in enumerate(self.linear_mean_layers):
            mu_s = mu_s + layer(x[i])
        
        sigma_s = 0
        for i,layer in enumerate(self.linear_sigma_layers):
            sigma_s = sigma_s + layer(x[i])

        x = torch.rand_like(mu_s)
        x = mu_s + x*sigma_s        
        return x
