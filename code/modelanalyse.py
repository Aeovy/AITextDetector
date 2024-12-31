import torch
import matplotlib.pyplot as plt
from Model import *
# 获取全连接层的权重
model=robertaLargeBiLSTMTextCNN2DCNN()
model.load_state_dict(torch.load('./model/robertaLargeBiLSTMTextCNN2DCNN.pth'))
for name, param in model.named_parameters():
    if "roberta" not in name:
        if param.grad is not None:
            gradients = param.grad.cpu().numpy()

            # 绘制梯度分布
            plt.hist(gradients.flatten(), bins=50)
            plt.title(f"Gradient Distribution of {name}")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.show()
for name, param in model.named_parameters():
    if "roberta" not in name  and "weight" in name:
        weights = param.data.cpu().numpy()
        sparsity = (abs(weights) < 1e-5).mean()
        print(f"Sparsity of {name}: {sparsity:.2%}")
for name, param in model.named_parameters():
    if "roberta" not in name  and "weight" in name:
        weights = param.data.cpu().numpy()

        # 绘制权重分布
        plt.hist(weights.flatten(), bins=50)
        plt.title(f"Weight Distribution of {name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.show()

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/trained_model_analysis")
for name, param in model.named_parameters():
    if "roberta" not in name :
        writer.add_histogram(f"Weights/{name}", param.data.cpu(), 0)
writer.close()