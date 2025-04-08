
# Transformer学习

transformer-v2.py 具备完整transformer结构（除交叉注意力模块）



## 附录

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)


## 文档

    1. 具备多头注意力，多层隐藏层
    2. 丢弃层、残差链接、层归一化避免过拟合
    3. 5000轮次，val_loss（交叉熵函数）到达1.9，能够输出英文
    4. 超参中device可以自行修改使用CUDA或CPU


## 环境变量

pip3 install torch torchvision torchaudio safetensors

