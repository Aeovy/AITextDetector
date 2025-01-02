from Model import robertaLargeBiLSTMTextCNN2DCNN_visualtest
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
import torch
# 初始化 TensorBoard 的 SummaryWriter
writer = SummaryWriter('tensorboard/model_visualization/test1')

# 创建模型实例
model = robertaLargeBiLSTMTextCNN2DCNN_visualtest()

# 定义一个模拟输入
dummy_input_ids = torch.randint(0, 200, (1, 512))  # 假设 vocab_size=30522, seq_len=128
dummy_attention_mask = torch.ones(1, 512)            # seq_len=128

# 将模型结构写入 TensorBoard
writer.add_graph(model, (dummy_input_ids, dummy_attention_mask))
writer.close()
# model=robertaLargeBiLSTMTextCNN2DCNN_visualtest()
# output = model(torch.randint(0, 200, (1, 512)), torch.ones(1, 512))
# dot = make_dot(output, params=dict(model.named_parameters()))
# dot.render("model_graph", format="pdf")  # 保存为 PNG 文件