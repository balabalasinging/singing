import time

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['STFangsong']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import jieba as jb
import re
from keras.models import load_model


# 定义删除除字母,数字，汉字以外的所有符号的函数
def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords



# 加载停用词
stopwords = stopwordslist("./data/stopwords.txt")
# 删除除字母,数字，汉字以外的所有符号
#df = pd.read_csv("./df.csv")
df = pd.read_csv("df.csv",encoding='utf-8',dtype=str)
df = df.astype(str)


# 设置最频繁使用的50000个词(在texts_to_matrix是会取前MAX_NB_WORDS,会取前MAX_NB_WORDS列)
MAX_NB_WORDS = 50000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 250
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

tokenizer.fit_on_texts(df['review'].values)
word_index = tokenizer.word_index
#print('共有 %s 个不相同的词语.' % len(word_index))


def predict(text):
    txt = remove_punctuation(text)
    txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

    model = load_model("saved_models/textcnn_best.path")
    #model.summary()
    pred = model.predict(padded)
    if pred > 0:
        print("积极")
    else:
        print("消极")
    


t = time.time()

print(predict('大堂不错，有四星的样子，房间的设施一般，感觉有点旧，卫生间细节不错，各种配套东西都不错，感觉还可以，有机会再去泰山还要入住。'))

print(predict('姐永远不卖了。坑爹伤不起。次奥。话说人家名字好吧。 好个性的名字草泥马奇葩，蒙牛姐永远不喝啦'))

print(predict('订单里明明有这本书为什么其它书都已收到为什么就没有这一本了？？？为什么了？？？'))

t1 = time.time()

t2 = t1-t
print(t2)




