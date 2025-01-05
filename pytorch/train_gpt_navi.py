import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
from models.custom_models.gpt.gpt_navi import GPTNaviConfig, GPTNaviModel, GPTNaviOutput, CausalLoss
from utils.count_params import count_parameters
from transformers import GPT2Tokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import random
from tqdm import tqdm
import copy


class DatasetBuilder(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.items = self.get_items()

    def get_items(self):
        df = pd.read_csv(self.csv_path)
        items = []
        for i in range(len(df)):
            s = df['raw'][i]
            s = json.loads(s)
            items.extend(s)

        random.shuffle(items)
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return item


train_dataset = DatasetBuilder(
    csv_path=r'D:\projects\dl_boilerplate\pytorch\datasets\flickr_annotations_30k.csv')
train_dataset = torch.utils.data.DataLoader(
    train_dataset, batch_size=4, shuffle=True)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

config = GPTNaviConfig()
model = GPTNaviModel(config=config)
model = model.train()

print("**"*20)
print(count_parameters(model))
print("**"*20)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(0, 100):
    print(f'Training Epoch {epoch+1}/100')
    p_bar = tqdm(train_dataset)
    for step, data in enumerate(p_bar):
        tokens = tokenizer(
            data,
            max_length=config.max_pos_emb_tokens,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        labels = copy.deepcopy(tokens).input_ids
        labels[labels == config.padding_idx] = -100

        model_output: GPTNaviOutput = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss_fn = CausalLoss()
        loss = loss_fn(labels=labels, logits=model_output.logits)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        p_bar.set_description(f'loss: {round(loss.item(),6)}')
