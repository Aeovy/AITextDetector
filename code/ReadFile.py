import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import Use_model_Functions 
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
data=None

def read_data(datapath):
    with open(datapath, 'r', encoding='utf-8') as file:
    #with open('./test_data.txt', 'r', encoding='utf-8') as file:
        data = file.readlines()

    ids = []
    labels = []
    contents = []
    classes={0:'人类编写',1:'大模型生成'}
    pattern = re.compile(r"'ID': '(\d+)', 'Label': '(\d+)', 'Content': '(.*?)'}")
    for line in data:
        match = pattern.search(line)
        if match:
            ids.append(match.group(1))
            labels.append(float(match.group(2)))
            contents.append(match.group(3).replace('\\n', ' ').replace('\\r', '').replace('\\u3000', ' '))
    return ids, labels, contents

_, labels, contents = read_data('./Data/train_data_32k.txt')
#简短文本量，用于测试
labels=labels#[0:100]
contents=contents#[0:100]



#划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(contents, labels, test_size=0.05, random_state=100)

#加载 BERT tokenizer
tokenizer = Use_model_Functions.tokenizer

#创建数据集和数据加载器
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=512)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=512)


        
