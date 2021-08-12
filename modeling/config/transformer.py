import torch
model_path = "../../saved_models/"  # 预训练模型路径
DEVICE = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
n_class = 2  # 分类数：分别为pos和neg
padding_idx = 0  # '_PAD_' token 对应的idx
learning_rate = 0.001 # 学习率
BATCH_SIZE = 128  # 训练批量
EPOCHS = 20  # 训练轮数
vocab_size = 20000  # 词汇表大小
hid_dim = 200  # embedding层隐向量维度
n_layers = 4   # EncoderLayer层数
n_heads = 8   # 多头自注意力head数
pf_dim = 256   # PositionwiseFeedFoward中间层的维度
dropout = 0.5  # dropout层
use_textcnn = True # 在transformer的输出上接上textcnn