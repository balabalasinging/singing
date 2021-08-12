import argparse
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.utils.data import TensorDataset, DataLoader

from modeling.model import TextCNN, TransformerEncoder, train, validation, test
from modeling.utils import build_word2id, build_word2vec, load_corpus, print_hyper_params

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model name")
args = parser.parse_args()
if args.model == "textcnn":
	from modeling.config.textcnn import *
elif args.model == "transformer":
	from modeling.config.transformer import *
else:
	raise NameError(f"Invalid model name: {args.model}")

# build dataloader
word2id = build_word2id(train_file='./Raw_Data/train.txt',
                        save_file='./Raw_Data/word2id.txt',
                        vocab_size=vocab_size)
vocab_size = len(word2id)
print('[INFO]: Vocabulary consists {} words.'.format(vocab_size))

print('[INFO]: Train Set Info.')
train_contents, train_labels = load_corpus('./Raw_Data/train.txt', word2id, max_sen_len=128)
train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.long),
                              torch.from_numpy(train_labels).type(torch.long))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
print('[INFO]: Valid Set Info.')
val_contents, val_labels = load_corpus('./Raw_Data/validation.txt', word2id, max_sen_len=128)
val_dataset = TensorDataset(torch.from_numpy(val_contents).type(torch.long),
                            torch.from_numpy(val_labels).type(torch.long))
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=0)
print('[INFO]: Test Set Info.')
test_contents, test_labels = load_corpus('./Raw_Data/test.txt', word2id, max_sen_len=128)
test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.long),
                             torch.from_numpy(test_labels).type(torch.long))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)


# build model
word2vec = build_word2vec('./dataset/wiki_word2vec_50.bin', word2id)
if args.model == "textcnn":
	print(f'[INFO]: Build Model: {args.model}')
	model = TextCNN(vocab_size, word2vec).to(DEVICE)
elif args.model == "transformer":
	print(f'[INFO]: Build Model: {args.model}+textcnn' if use_textcnn else f'[INFO]: Build Model: {args.model}')
	model = TransformerEncoder(vocab_size=vocab_size,
	                           pad_idx=padding_idx,
	                           hid_dim=hid_dim,
	                           n_layers=n_layers,
	                           n_heads=n_heads,
	                           pf_dim=pf_dim,
	                           dropout=dropout,
	                           use_textcnn=use_textcnn,
	                           device=DEVICE).to(DEVICE)




'''
if model_path:
    model.load_state_dict(torch.load(model_path))
model.to(DEVICE)
'''

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# 设置损失函数
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5)

print('\n-------------------------- hyperparameters --------------------------------')
print_hyper_params(args.model)
print('\n')

train_losses = []
train_acces = []
val_losses = []
val_acces = []
best_val_acc = 0
best_model_pth = model_path + args.model + '_best.pth'

print('----------------- [INFO]: Start Train and valid ------------------')
for epoch in range(1, EPOCHS + 1):
	tr_loss, tr_acc = train(model, args.model, train_dataloader, epoch, optimizer, criterion, scheduler)
	val_loss, val_acc = validation(model, args.model, val_dataloader, epoch, criterion)
	if val_acc > best_val_acc:
		best_val_acc = val_acc
		torch.save(model.state_dict(), best_model_pth)
	train_losses.append(tr_loss)
	train_acces.append(tr_acc)
	val_losses.append(val_loss)
	val_acces.append(val_acc)

print('------------------ [INFO]: Start Test ------------------------')
test(model, args.model, best_model_pth, test_dataloader)


