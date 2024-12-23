from PIL import Image
import numpy as np
import torch

class Batch:
    def __init__(self, category, input_ids, special_token_mask, attention_mask, pixel_values, labels):
        self.category = category
        self.input_ids = input_ids
        self.special_token_mask = special_token_mask
        self.attention_mask = attention_mask
        self.pixel_values = pixel_values
        self.labels = labels

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.special_token_mask = self.special_token_mask.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.pixel_values = self.pixel_values.to(device)

class MeeshoDataloader:
    def __init__(self, df, cat_info, batch_size, img_processor):
        self.df = df
        self.cat_info = cat_info
        self.bs = batch_size
        self.img_processor = img_processor

    def __len__(self):
        return sum((len(idxs)+self.bs-1)//self.bs for idxs in self.group2idxs.values())

    def _sample(self):
        group2idxs = self.df.groupby("Category").apply(lambda group: group.index.tolist())
        for cat in np.random.permutation(group2idxs.index):
            idxs = group2idxs[cat]
            for i in range(0, len(idxs), self.bs):
                yield idxs[i:i+self.bs]

    def __iter__(self):
        for idxs in self._sample():
            cat = self.df.loc[idxs[0], 'Category']
            
            input_ids = self.cat_info.loc[cat,'input_ids']
            special_token_mask = self.cat_info.loc[cat,'special_token_mask']
            attention_mask = self.cat_info.loc[cat,'attention_mask']
            
            images = [Image.open(path) for path in self.df.loc[idxs,'img_path']]
            pixel_values = self.img_processor(images=images, return_tensors='pt', size=(224,224))
            
            labels = self.df.loc[idxs,'labels'] if 'labels' in self.df.columns else None
            yield Batch(category=cat, input_ids=torch.tensor(input_ids),
                        special_token_mask=torch.tensor(special_token_mask),
                        attention_mask=torch.tensor(attention_mask),
                        pixel_values=pixel_values.pixel_values,
                        labels=torch.tensor(labels.tolist()))