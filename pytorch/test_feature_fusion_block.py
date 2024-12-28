import torch
from models import FeatureFusionBlock

in_features_list = [12,24,36]
out_features = 48

feature_fusion = FeatureFusionBlock(
    in_features_list=in_features_list,
    out_features=out_features
)

x = [
    torch.randn(1,i) for i in in_features_list
]

fused_feature = feature_fusion(x)
print(fused_feature.shape)