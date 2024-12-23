import pandas as pd

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