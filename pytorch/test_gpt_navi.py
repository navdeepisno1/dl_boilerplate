from models.custom_models.gpt.gpt_navi import GPTNaviConfig,GPTNaviModel,GPTNaviOutput
from utils.count_params import count_parameters
from transformers import GPT2Tokenizer 

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPTNaviConfig()
model = GPTNaviModel(config=config)

tokens = tokenizer(["Hello I am navdeep"],max_length=config.max_pos_emb_tokens,padding="max_length",return_tensors="pt")
input_ids = tokens.input_ids
attention_mask = tokens.attention_mask
print("**"*20)
print(count_parameters(model))
print("**"*20)

model_output:GPTNaviOutput = model(
    input_ids=input_ids,
    attention_mask = attention_mask
)



print(model_output.logits.shape)