from torch.utils.data import Dataset

class DatasetBuilder(Dataset):
    def __init__(self):
        super().__init__()
        self.items = self.get_items()

    def get_items(self):
        return []
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return item
        
