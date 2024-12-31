from Model import *
from ReadFile import *
from sklearn.metrics import accuracy_score,f1_score
from torch.utils.tensorboard import SummaryWriter
if __name__ == '__main__':
    writer=SummaryWriter("model_visualization/test1")
    model=robertaLargeBiLSTMTextCNN2DCNN()
    model.load_state_dict(torch.load('./model/robertaLargeBiLSTMTextCNN2DCNN_epoch1.pth'))
    model.eval()
    model.to('cpu')
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,num_workers=4)
    val_labels = []
    val_preds = []
    with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']

                attention_mask=attention_mask.to('cpu')
                input_ids=input_ids.to('cpu')
                labels=labels.to('cpu')
                outputs = model(input_ids, attention_mask)
                writer.add_graph(model, (input_ids, attention_mask))
                outputs = outputs.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                for i in outputs:
                    if i>=0.5:
                        val_preds.append(1)
                    else:
                        val_preds.append(0)
                for i in labels:
                    val_labels.append(i)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds)
            print(f'Validation Accuracy: {val_acc:.4f}, Validation MarcoF1:{val_f1:.4f}')
    