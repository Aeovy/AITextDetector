import torch
import numpy as np
from transformers import AutoTokenizer
from Model import CNNModel1,robertaModel,robertaModelLarge,robertaBiLSTMTextCNN,robertaLargeBiLSTMTextCNN,robertaLargeBiLSTMTextCNN2DCNN,device
tokenizer=AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large',clean_up_tokenization_spaces=True,local_files_only=True)
def encoding(text,tokenizer=tokenizer):
    tokenizer = tokenizer
    coding = tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, return_token_type_ids=False, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
    output_ids = coding['input_ids'].flatten().unsqueeze(0).to(device)
    output_attention_mask = coding['attention_mask'].flatten().unsqueeze(0).to(device)
    return output_ids, output_attention_mask
def load_model(model_name='robertaLargeBiLSTMTextCNN'):
    modelpath=None
    if model_name == 'CNNModel1':
        model = CNNModel1()
        modelpath='./model/CNNModel1.pth'
    elif model_name == 'robertaModel':
        model = robertaModel()
        modelpath='./model/roberta.pth'
    elif model_name == 'robertaModelLarge':
        model = robertaModelLarge()
        modelpath='./model/robertaLarge.pth'
    elif model_name == 'robertaBiLSTMTextCNN':
        model = robertaBiLSTMTextCNN()
        modelpath='./model/robertaBiLSTMTextCNN.pth'
    elif model_name == 'robertaLargeBiLSTMTextCNN':
        model = robertaLargeBiLSTMTextCNN()
        modelpath='./model/robertaLargeBiLSTMTextCNN.pth'
    elif model_name == 'robertaLargeBiLSTMTextCNN2DCNN':
        model = robertaLargeBiLSTMTextCNN2DCNN()
        modelpath='./model/robertaLargeBiLSTMTextCNN2DCNN.pth'
    model.load_state_dict(torch.load(modelpath,weights_only=False,map_location=device))
    model.to(device)
    model.eval()
    return model
