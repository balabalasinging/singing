import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.config import textcnn, transformer


class Model(nn.Module):
	def __init__(self, vocab_size):
		super(Model, self).__init__()
		self.vocab_size = vocab_size
		self.n_class = 2
	
	def forward(self, x):
		pass

class TextCNN(Model):
    def __init__(self, vocab_size, word2vec):
        super(TextCNN, self).__init__(vocab_size)
        self.embedding_dim = 50
        self.drop_keep_prob = 0.5
        self.pretrained_embed = word2vec
        self.kernel_num = textcnn.kernel_num
        self.kernel_size = textcnn.kernel_size

        # 使用预训练的词向量
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.embedding.weight.data.copy_(torch.from_numpy(self.pretrained_embed))
        # self.embedding.weight.requires_grad = True
        # 卷积层
        self.conv1 = nn.Conv2d(1, textcnn.kernel_num, (textcnn.kernel_size[0], self.embedding_dim))
        self.conv2 = nn.Conv2d(1, textcnn.kernel_num, (textcnn.kernel_size[1], self.embedding_dim))
        self.conv3 = nn.Conv2d(1, textcnn.kernel_num, (textcnn.kernel_size[2], self.embedding_dim))
        # Dropout
        self.dropout = nn.Dropout(self.drop_keep_prob)
        # 全连接层
        self.fc = nn.Linear(len(self.kernel_size) * self.kernel_num, self.n_class)

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length, embedding_dim)
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        x = x.to(torch.int64)
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv2)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv3)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x



class TransformerEncoder(Model):
	def __init__(self,
	             vocab_size,
	             pad_idx,
	             hid_dim,  # embedding之后词向量的维度
	             n_layers,  # num of EncoderLayer
	             n_heads,  # num of head in MultiHeadAttentionLayer
	             pf_dim,  # pf: positionwise feed forward
	             dropout,
	             use_textcnn,
	             device,
	             max_length=128):
		super(TransformerEncoder, self).__init__(vocab_size)
		self.device = device
		self.use_textcnn = use_textcnn
		self.tok_embedding = nn.Embedding(vocab_size, hid_dim)
		self.pos_embedding = nn.Embedding(max_length, hid_dim)
		self.layers = nn.ModuleList([EncoderLayer(hid_dim,
		                                          n_heads,
		                                          pf_dim,
		                                          dropout,
		                                          device)
		                             for _ in range(n_layers)])
		self.dropout = nn.Dropout(dropout)
		self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
		self.pad_idx = pad_idx
		# 全连接层
		self.fc = nn.Linear(hid_dim, self.n_class)
		if self.use_textcnn:
			# 卷积层
			self.conv1 = nn.Conv2d(1, textcnn.kernel_num, (textcnn.kernel_size[0], hid_dim))
			self.conv2 = nn.Conv2d(1, textcnn.kernel_num, (textcnn.kernel_size[1], hid_dim))
			self.conv3 = nn.Conv2d(1, textcnn.kernel_num, (textcnn.kernel_size[2], hid_dim))
			# 全连接层
			self.fc = nn.Linear(len(textcnn.kernel_size) * textcnn.kernel_num, self.n_class)
	
	def make_src_mask(self, src):
		# src = [batch_size, src_len]
		src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
		# src_mask = [batch_size, 1, 1, src_len]
		
		return src_mask
	
	@staticmethod
	def conv_and_pool(x, conv):
		# x: (batch, 1, sentence_length, hid_dim)
		x = conv(x)
		# x: (batch, kernel_num, H_out, 1)
		x = F.relu(x.squeeze(3))
		# x: (batch, kernel_num, H_out)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)
		#  (batch, kernel_num)
		return x
	
	def forward(self, src):
		# src = [batch_size, src_len]
		src_mask = self.make_src_mask(src)
		# src_mask = [batch_size, 1, 1, src_len]
		batch_size = src.shape[0]
		src_len = src.shape[1]
		
		pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
		# pos = [batch_size, src_len]
		src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
		# src = [batch_size, src_len, hid_dim]
		
		for layer in self.layers:
			src = layer(src, src_mask)
		# src = [batch_size, src_len, hid_dim]
		if self.use_textcnn:
			x = src.unsqueeze(1)
			x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
			x2 = self.conv_and_pool(x, self.conv2)  # (batch, kernel_num)
			x3 = self.conv_and_pool(x, self.conv3)  # (batch, kernel_num)
			x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
			x = self.dropout(x)
			x = self.fc(x)
			x = F.log_softmax(x, dim=1)
		else:
			x = torch.mean(src, 1)
			# src = [batch_size, hid_dim]
			x = self.dropout(x)
			x = self.fc(x)
			x = F.log_softmax(x, dim=1)
		return x


class EncoderLayer(nn.Module):
	def __init__(self,
	             hid_dim,
	             n_heads,
	             pf_dim,
	             dropout,
	             device):
		super().__init__()
		
		self.self_attn_layer_norm = nn.LayerNorm(hid_dim)  # layer normalization for self-attention layer
		self.ff_layer_norm = nn.LayerNorm(hid_dim)  # layer normalization for feed-forward layer
		self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
		self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
		                                                             pf_dim,
		                                                             dropout)
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, src, src_mask):
		# src = [batch_size, src_len, hid_dim]
		# src_mask = [batch_size, 1, 1, src_len]
		
		# self attention
		_src, _ = self.self_attention(src, src, src, src_mask)  # query = key = value = src
		
		# dropout, residual connection and layer norm
		src = self.self_attn_layer_norm(src + self.dropout(_src))
		# src = [batch_size, src_len, hid_dim]
		
		# positionwise feedforward
		_src = self.positionwise_feedforward(src)
		
		# dropout, residual and layer norm
		src = self.ff_layer_norm(src + self.dropout(_src))
		# src = [batch_size, src_len, hid_dim]
		
		return src


class MultiHeadAttentionLayer(nn.Module):
	def __init__(self, hid_dim, n_heads, dropout, device):
		super().__init__()
		
		assert hid_dim % n_heads == 0
		
		self.hid_dim = hid_dim
		self.n_heads = n_heads
		self.head_dim = hid_dim // n_heads
		
		self.fc_q = nn.Linear(hid_dim, hid_dim)
		self.fc_k = nn.Linear(hid_dim, hid_dim)
		self.fc_v = nn.Linear(hid_dim, hid_dim)
		
		self.fc_o = nn.Linear(hid_dim, hid_dim)
		
		self.dropout = nn.Dropout(dropout)
		
		self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
	
	def forward(self, query, key, value, mask=None):
		# query = [batch_size, query_len, hid_dim]
		# key = [batch_size, key_len, hid_dim]
		# value = [batch_size, value_len, hid_dim]
		batch_size = query.shape[0]
		
		Q = self.fc_q(query)
		K = self.fc_k(key)
		V = self.fc_v(value)
		# Q = [batch_size, query_len, hid_dim]
		# K = [batch_size, key_len, hid_dim]
		# V = [batch_size, value_len, hid_dim]
		
		Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
		# Q = [batch_size, n_heads, query_len, head_dim]
		# K = [batch_size, n_heads, key_len, head_dim]
		# V = [batch_size, n_heads, value_len, head_dim]
		
		energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
		# energy = [batch_size, n_heads, query_len, key_len]
		
		if mask is not None:
			energy = energy.masked_fill(mask == 0, -1e10)
		
		attention = torch.softmax(energy, dim=-1)
		# attention = [batch_size, n_heads, query_len, key_len]
		x = torch.matmul(self.dropout(attention), V)
		# x = [batch_size, n_heads, query_len, head_dim]
		x = x.permute(0, 2, 1, 3).contiguous()
		# x = [batch_size, query_len, n_heads, head_dim]
		x = x.view(batch_size, -1, self.hid_dim)
		# x = [batch_size, query_len, hid_dim]
		x = self.fc_o(x)
		# x = [batch_size, query_len, hid_dim]
		
		return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
	def __init__(self, hid_dim, pf_dim, dropout):
		super().__init__()
		
		self.fc_1 = nn.Linear(hid_dim, pf_dim)
		self.fc_2 = nn.Linear(pf_dim, hid_dim)
		self.gelu = nn.GELU()
		self.dropout = nn.Dropout(dropout)
	
	def forward(self, x):
		# x = [batch_size, seq_len, hid_dim]
		# x = self.dropout(torch.relu(self.fc_1(x)))
		# use GELU activation function
		
		x = self.dropout(self.gelu(self.fc_1(x)))
		# x = [batch_size, seq_len, pf_dim]
		x = self.fc_2(x)
		# x = [batch_size, seq_len, hid_dim]
		
		return x


def train(model, model_name, dataloader, epoch, optimizer, criterion, scheduler):
	if model_name == "textcnn":
		BATCH_SIZE, DEVICE = textcnn.BATCH_SIZE, textcnn.DEVICE
	elif model_name == "transformer":
		BATCH_SIZE, DEVICE = transformer.BATCH_SIZE, transformer.DEVICE
	# 定义训练过程
	train_loss, train_acc = 0.0, 0.0
	count, correct = 0, 0
	for batch_idx, (x, y) in enumerate(dataloader):
		x, y = x.to(DEVICE), y.to(DEVICE)
		optimizer.zero_grad()
		output = model(x)
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		correct += (output.argmax(1) == y).float().sum().item()
		count += len(x)
		"""
		if (batch_idx + 1) % 10 == 0:
			print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
				epoch, batch_idx * len(x), len(dataloader.dataset),
				100. * batch_idx / len(dataloader), loss.item()))
		"""
	
	train_loss *= BATCH_SIZE
	train_loss /= len(dataloader.dataset)
	train_acc = correct / count
	print('train epoch: {}\taverage loss: {:.6f}\taccuracy:{:.4f}%'.format(epoch, train_loss, 100. * train_acc))
	scheduler.step()
	
	return train_loss, train_acc


def validation(model, model_name, dataloader, epoch, criterion):
	if model_name == "textcnn":
		BATCH_SIZE, DEVICE = textcnn.BATCH_SIZE, textcnn.DEVICE
	elif model_name == "transformer":
		BATCH_SIZE, DEVICE = transformer.BATCH_SIZE, transformer.DEVICE
	model.eval()
	# 验证过程
	val_loss, val_acc = 0.0, 0.0
	count, correct = 0, 0
	for _, (x, y) in enumerate(dataloader):
		x, y = x.to(DEVICE), y.to(DEVICE)
		output = model(x)
		loss = criterion(output, y)
		val_loss += loss.item()
		correct += (output.argmax(1) == y).float().sum().item()
		count += len(x)
	
	val_loss *= BATCH_SIZE
	val_loss /= len(dataloader.dataset)
	val_acc = correct / count
	# 打印准确率
	print(
		'validation:train epoch: {}\taverage loss: {:.6f}\t accuracy:{:.2f}%'.format(epoch, val_loss, 100 * val_acc))
	
	return val_loss, val_acc


def test(model, model_name, best_model_pth, dataloader):
	if model_name == "textcnn":
		DEVICE = textcnn.DEVICE
	elif model_name == "transformer":
		DEVICE = transformer.DEVICE
	model.load_state_dict(torch.load(best_model_pth))
	model.eval()
	model.to(DEVICE)
	
	# 测试过程
	count, correct = 0, 0
	for _, (x, y) in enumerate(dataloader):
		x, y = x.to(DEVICE), y.to(DEVICE)
		output = model(x)
		correct += (output.argmax(1) == y).float().sum().item()
		count += len(x)
	
	# 打印准确率
	print('test accuracy:{:.2f}%.'.format(100 * correct / count))
	
	
	
