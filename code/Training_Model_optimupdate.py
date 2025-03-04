from sklearn.metrics import accuracy_score,f1_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel,AutoTokenizer
from tqdm import tqdm
from Model import *
from ReadFile import *
if __name__ == '__main__':
    modelname='robertaLargeBiLSTMTextCNN2DCNN'
    writer=SummaryWriter("tensorboard/{0}/test2".format(modelname))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,prefetch_factor=2,num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,num_workers=4)
    model=robertaLargeBiLSTMTextCNN2DCNN()
    #model=robertaLargeBiLSTMTextCNN2DCNN()
    #model.load_state_dict(torch.load('./model/robertaLargeBiLSTMTextCNN2DCNN_epoch1.pth',map_location=device))
    for name, param in model.named_parameters():
        if "roberta" in name:
            param.requires_grad = False
    model.to(device)

    criterion = nn.BCELoss()
    #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.5))

    # optimizer=optim.AdamW(filter(lambda p:p.requires_grad,model.parameters()), lr=1e-6,weight_decay=1e-4)
    optimizer = optim.AdamW([
    {'params': model.bilstm.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': model.convs.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
    {'params': model.conv2d.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
    {'params': model.conv2d_2.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
    {'params': model.fc2.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3}
    ])
    #学习率调度器测试
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=0.1,patience=1)
    #scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)

    print('Training')
    num_epochs = 3
    steps=0
    schedulersteps=0
    loss_sum=0
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label'].float()
            labels=labels.to(device)
            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps+=1
            writer.add_scalar('Loss/train', loss, steps)
            if epoch>=0:
                schedulersteps+=1
                if schedulersteps%200==0:
                    model.eval()
                    val_preds = []
                    val_labels = []
                    with torch.no_grad():
                        for batch in val_loader:
                            input_ids = batch['input_ids']
                            attention_mask = batch['attention_mask']
                            labels = batch['label']
                            attention_mask=attention_mask.to(device)
                            input_ids=input_ids.to(device)
                            labels=labels.to(device)
                            outputs = model(input_ids, attention_mask)
                            outputs = outputs.detach().cpu().numpy()
                            labels = labels.detach().cpu().numpy()
                            for i in outputs:
                                if i>=0.5:
                                    val_preds.append(1)
                                else:
                                    val_preds.append(0)
                            for i in labels:
                                val_labels.append(i)
                    val_f1 = round(f1_score(val_labels, val_preds),3)
                    model.train()
                    scheduler.step(val_f1)
                    writer.add_scalar('f1/steps', val_f1, steps)
                    writer.add_scalar('lr/steps',scheduler._last_lr[0],steps)
                    schedulersteps=0
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                attention_mask=attention_mask.to(device)
                input_ids=input_ids.to(device)
                labels=labels.to(device)
                outputs = model(input_ids, attention_mask)
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                for i in outputs:
                    if i>=0.5:
                        val_preds.append(1)
                    else:
                        val_preds.append(0)
                for i in labels:
                    val_labels.append(i)
        #scheduler.step()
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('MarcoF1/val', val_f1, epoch)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}, Validation MarcoF1:{val_f1:.4f}')
        torch.save(model.state_dict(), './model/{0}_epoch{1}.pth'.format(modelname,epoch+1))
    writer.close()  
    torch.save(model.state_dict(), './model/{0}.pth'.format(modelname))