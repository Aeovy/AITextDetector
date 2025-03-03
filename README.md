# AITextDetector

## 项目简介

本项目基于RoBERTa模型结合BiLSTM、TextCNN和2DCNN，用于检测一段文本是否由AI生成。

## 主要特性

- 使用TensorBoard可视化模型训练与评估
- Docker容器化部署

## 文件结构

```
Competition_LLM_CLS/
├── code/
│   ├── Model.py              # 模型定义
│   ├── Training_Model_optimupdate.py  # 模型训练
│   └── Use_Model_By_Input.py # 通过命令行使用模型
│   └── ReadFile.py           # 数据加载处理
├── Dockerfile                # Docker配置
├── requirements.txt          # Python依赖
├── License                   
└── README.md                 
```


## 模型架构

本项目主要使用 **robertaLargeBiLSTMTextCNN2DCNN** ，其他模型仅作为测试和比较用途。


## 模型训练

模型训练通过 [`code/Training_Model_optimupdate.py`](code/Training_Model_optimupdate.py ) 完成。

## 模型评估

训练过程中会在验证集上评估模型性能，计算正确率和F1。所有的训练日志和指标都会记录在TensorBoard中，可以打开tensorboard查看。

## Docker部署

项目提供了Docker部署支持，可以通过以下命令构建和运行容器：


# 构建Docker镜像
docker build -t llm-cls-roberta .
Docker构建时会预载RoBERTa-large模型。

# 运行容器
docker run -it -p 5000:5000 llm-cls-roberta python app.py


## License
请勿将本项目不经修改直接作为课程结课作业
This project is licensed under the GPLv3 License - see the [LICENSE](./LICENSE) file for details.
