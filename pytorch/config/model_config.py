from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)