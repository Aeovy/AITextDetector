import torch
import torch.nn as nn
from transformers import BertModel,AutoModel

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=512, kernel_size=128, padding=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=64, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 * 35, 64) 
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.permute(0, 2, 1)
        #####
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x

class robertaModel(nn.Module):
    def __init__(self):
        super(robertaModel, self).__init__()
        self.roberta = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=256, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128*256, 64) 
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x=output.last_hidden_state.permute(0, 2, 1)
        x = self.conv1(x)
  
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)

        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x
class robertaBiLSTMTextCNN(nn.Module):
    def __init__(self, roberta_model_name='hfl/chinese-roberta-wwm-ext', dropout_prob=0.3):
        super(robertaBiLSTMTextCNN, self).__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name,local_files_only=True)
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,  
            hidden_size=512,                             
            num_layers=1,                                
            batch_first=True,                            
            bidirectional=True#双向LSTM
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1024,#双向LSTM输出维度(512*2)
                out_channels=512,    
                kernel_size=3,       
                padding=1
            ),
            nn.Conv1d(
                in_channels=1024,
                out_channels=512,
                kernel_size=5,
                padding=2
            ),
            nn.Conv1d(
                in_channels=1024,
                out_channels=512,
                kernel_size=7,
                padding=3
            )
        ])
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(256 * 3 * 512, 256)  #512通道/2* 3个卷积核 * 512 seq长度
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        
        #BiLSTM
        lstm_out, _ = self.bilstm(last_hidden_state)#[batch_size, seq_len, 2*hidden_size]
        x = lstm_out.permute(0, 2, 1)#[batch_size, 512, seq_len]
        #TextCNN
        conv_features = []
        for conv in self.convs:
            conv_x = conv(x)#[batch_size, out_channels, seq_len]
            conv_x = self.relu(conv_x)
            conv_x=conv_x.permute(0, 1, 2)#[batch_size, seq_len, out_channels]
            pooled_x = self.pool(conv_x)
            pooled_x=pooled_x.permute(0, 2, 1)#[batch_size, out_channels/2, seq_len]
            conv_features.append(pooled_x)
        
        #拼接所有卷积核的输出
        x = torch.cat(conv_features, dim=1)  #[batch_size, out_channels * num_convs, seq_len]
        x = x.view(x.size(0), -1)            #[batch_size, 256 * 3 * seq_len]
        
        #全连接层
        x = self.dropout(x)
        x = self.fc(x)                       #[batch_size, 64]
        x = self.fc2(x)                      #[batch_size, 1]
        x = self.sigmoid(x)                  #[batch_size, 1]
        x = x.squeeze(1)                     #[batch_size]
        return x
class robertaLargeBiLSTMTextCNN(nn.Module):
    def __init__(self, roberta_model_name='hfl/chinese-roberta-wwm-ext-large', dropout_prob=0.3):
        super(robertaLargeBiLSTMTextCNN, self).__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name,local_files_only=True)
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,  
            hidden_size=512,                             
            num_layers=1,                                
            batch_first=True,                            
            bidirectional=True#双向LSTM
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1024,#双向LSTM输出维度(512*2)
                out_channels=512,    
                kernel_size=3,       
                padding=1
            ),
            nn.Conv1d(
                in_channels=1024,
                out_channels=512,
                kernel_size=5,
                padding=2
            ),
            nn.Conv1d(
                in_channels=1024,
                out_channels=512,
                kernel_size=7,
                padding=3
            )
        ])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(512 * 3 * 256, 256)  #512通道* 3个卷积核 * 512 seq长度/2
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        #BiLSTM
        lstm_out, _ = self.bilstm(last_hidden_state)#[batch_size, seq_len, 2*hidden_size]
        x = lstm_out.permute(0, 2, 1)#[batch_size, 512, seq_len]
        #TextCNN
        conv_features = []
        for conv in self.convs:
            conv_x = conv(x)#[batch_size, out_channels, seq_len]
            conv_x = self.relu(conv_x)
            pooled_x = self.pool(conv_x)#[batch_size, out_channels, seq_len/2]
            conv_features.append(pooled_x)
        #拼接所有卷积核的输出
        x = torch.cat(conv_features, dim=1)  #[batch_size, out_channels * num_convs, seq_len]
        x = x.view(x.size(0), -1)            #[batch_size, 256 * 3 * seq_len]
        #全连接层
        x = self.dropout(x)
        x = self.fc(x)                       #[batch_size, 64]
        x = self.fc2(x)                      #[batch_size, 1]
        x = self.sigmoid(x)                  #[batch_size, 1]
        x = x.squeeze(1)                     #[batch_size]
        return x
class robertaLargeBiLSTMTextCNN2DCNN(nn.Module):
    def __init__(self, roberta_model_name='hfl/chinese-roberta-wwm-ext-large', dropout_prob=0.3):
        super(robertaLargeBiLSTMTextCNN2DCNN, self).__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name,local_files_only=True)
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,  
            hidden_size=512,                             
            num_layers=1,                                
            batch_first=True,                            
            bidirectional=True#双向LSTM
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=1024,#双向LSTM输出维度(512*2)
                out_channels=512,    
                kernel_size=3,       
                padding=1
            ),
            nn.Conv1d(
                in_channels=1024,
                out_channels=512,
                kernel_size=5,
                padding=2
            ),
            nn.Conv1d(
                in_channels=1024,
                out_channels=512,
                kernel_size=7,
                padding=3
            )
        ])
        self.conv2d = nn.Conv2d(
                in_channels=3,
                out_channels=128,    
                kernel_size=(3,3),       
                padding=1
            )
        self.conv2d_2 = nn.Conv2d(
                in_channels=128,
                out_channels=32,    
                kernel_size=(3,3),       
                padding=1
            )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.pool2d=nn.MaxPool2d(kernel_size=(2,2))
        self.fc = nn.Linear(32 * 64 * 128, 256)  #32通道* 64词向量维度 * 128 seq长度
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        #BiLSTM
        lstm_out, _ = self.bilstm(last_hidden_state)#[batch_size, seq_len, 2*hidden_size]
        x = lstm_out.permute(0, 2, 1)#[batch_size, dim, seq_len]
        #CNN
        conv_features = []
        for conv in self.convs:
            conv_x = conv(x)#[batch_size, out_channels, seq_len]
            conv_x = self.relu(conv_x)
            pooled_x = self.pool(conv_x)#[batch_size, out_channels, seq_len/2]
            pooled_x=pooled_x.permute(0, 2, 1).unsqueeze(1)#[batch_size, seq_len/2, out_channels]
            conv_features.append(pooled_x)
        #拼接所有卷积核的输出
        x = torch.cat(conv_features, dim=1)  #[batch_size, 3,out_channels * num_convs, seq_len]
        x=self.conv2d(x)
        x=self.relu(x)
        x=self.pool2d(x)
        x=self.conv2d_2(x)
        x=self.pool2d(x)
        x = x.view(x.size(0), -1)            #[batch_size, 256 * 3 * seq_len]
        #全连接层
        x = self.dropout(x)
        x = self.fc(x)                       #[batch_size, 64]
        x = self.fc2(x)                      #[batch_size, 1]
        x = self.sigmoid(x)                  #[batch_size, 1]
        x = x.squeeze(1)                     #[batch_size]
        return x
class robertaModelLarge(nn.Module):
    def __init__(self):
        super(robertaModelLarge, self).__init__()
        self.roberta = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext-large',local_files_only=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024*512, 256) 
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state
        x = output.view(output.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)
        return x