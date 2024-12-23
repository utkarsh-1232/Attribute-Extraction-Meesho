from PIL import Image
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch

def add_img_path(id_, path, img_folder):
    return str(path/f'{img_folder}/{str(id_).zfill(6)}.jpg')

class LabelEncoder:
    def __init__(self, df):
        self.vocab = df.drop(columns=['id','len']).groupby('Category').agg(lambda col: col.unique())
        self.label2id_map, self.id2label_map = {}, {}
        for cat, row in self.vocab.iterrows():
            self.label2id_map[cat] = {
                attr:{str(label):idx for idx, label in enumerate(labels)}
                for attr, labels in row.items()
            }
            self.id2label_map[cat] = {
                attr:{idx:str(label) for idx, label in enumerate(labels)}
                for attr, labels in row.items()
            }
    
    def label2id(self, cat, attr,  label):
        return self.label2id_map[cat][attr][label]

    def id2label(self, cat, attr, id_):
        return self.id2label_map[cat][attr][id_]

    def num_classes(self, cat):
        return [len(labels) for labels in self.vocab.loc[cat].values if len(labels)>1]

def get_labels(row, encoder):
    cat = row['Category']
    num_attrs = row['len']
    ids = []
    for i in range(num_attrs):
        attr = f'attr_{i+1}'
        id_ = encoder.label2id(cat, attr, str(row[attr]))
        ids.append(id_)
    return tuple(ids)

def process_df(df, path, encoder, is_test_df=False):
    df = df.copy()
    img_folder = 'test_images' if is_test_df else 'train_images'
    df['img_path'] = df['id'].apply(add_img_path, path=path, img_folder=img_folder)
    if is_test_df: return df
    df['labels'] = df.apply(get_labels, axis=1, encoder=encoder)
    return df

def add_question_and_tokenize(row, tokenizer, encode_token_id):
    cat = row.name
    attrs = row['Attribute_list']
    question_lines = [f'For this image of {cat[:-1]}, please answer the following:']
    question_lines.extend([f'[Encode] What is the {attr}?' for attr in attrs])
    question = '\n'.join(question_lines)
    
    tokenized = tokenizer(question, padding=True, truncation=True, return_tensors='pt')
    tokenized['special_token_mask'] = (tokenized['input_ids']==encode_token_id).squeeze()
    tokenized = {k:v.tolist() for k, v in tokenized.items()}

    return pd.Series({**row.to_dict(), 'question':question, **tokenized})

def process_cat_info(cat_info, tokenizer):
    cat_info = cat_info.copy()
    tokenizer.add_special_tokens({'additional_special_tokens':['[Encode]']})
    encode_token_id = tokenizer.convert_tokens_to_ids('[Encode]')
    cat_info.set_index('Category', inplace=True)
    return cat_info.apply(add_question_and_tokenize, axis=1, args=(tokenizer, encode_token_id))

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
            pixel_values = self.img_processor(images=images, return_tensors='pt', size=(224,224))
            
            labels = self.df.loc[idxs,'labels'] if 'labels' in self.df.columns else None
            yield MiniBatch(category=cat, input_ids=torch.tensor(input_ids),
                            special_token_mask=torch.tensor(special_token_mask),
                            attention_mask=torch.tensor(attention_mask),
                            pixel_values=pixel_values.pixel_values,
                            labels=torch.tensor(labels.tolist()))