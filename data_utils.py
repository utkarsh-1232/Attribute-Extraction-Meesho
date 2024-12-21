from PIL import Image
from dataclasses import dataclass
import numpy as np
import torch

class LabelEncoder:
    def __init__(self, df):
        attr_cols = [col for col in df.columns if col.startswith('attr')]
        self.unique_labels = df.groupby('Category')[attr_cols].agg(lambda col: col.unique())
        self.label2id_map = {}
        self.id2label_map = {}
        for category, row in self.unique_labels.iterrows():
            self.label2id_map[category] = {
                attr:{str(label):idx for idx, label in enumerate(labels)}
                for attr, labels in row.items()
            }
            self.id2label_map[category] = {
                attr:{idx:str(label) for idx, label in enumerate(labels)}
                for attr, labels in row.items()
            }

    def label2id(self, category, attr,  label):
        return self.label2id_map[category][attr][label]

    def id2label(self, category, attr, id_):
        return self.id2label_map[category][attr][id_]

@dataclass
class MiniBatch:
    category: str
    input_ids: torch.FloatTensor
    special_token_mask: torch.FloatTensor
    attention_mask: torch.FloatTensor
    pixel_values: torch.FloatTensor
    labels: torch.FloatTensor

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
            pixel_values = self.img_processor(images=images, return_tensor='pt', size=(224,224))
            
            labels = self.df.loc[idxs,'labels']
            yield MiniBatch(category=cat, input_ids=torch.tensor(input_ids),
                            special_token_mask=torch.tensor(special_token_mask),
                            attention_mask=torch.tensor(attention_mask),
                            pixel_values=pixel_values, labels=torch.tensor(labels))