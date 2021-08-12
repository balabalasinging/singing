from collections import Counter

import gensim
import numpy as np


def cat_to_id(classes=None):
    """
    :param classes: 分类标签；默认为1:pos, 0:neg
    :return: {分类标签：id}
    """
    if not classes:
        classes = ['0', '1']
    cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
    return classes, cat2id


def build_word2id(train_file, save_file, vocab_size, save_to_path=None):
    """
    :param save_file: word2id保存地址
    :param save_to_path: 保存训练语料库中的词组对应的word2id到本地
    :return: None
    """
    word2id = {'_PAD_': 0, "_UNK_": 1}
    path = train_file
    word2count = {}

    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip('"').split()
            for word in sp[1:]:
                if word not in word2id.keys():
                    word2count[word] = 1
                else:
                    word2count[word] += 1

    words_freq = sorted(word2count.items(), key=lambda item: item[1], reverse=True)
    if len(word2count) > vocab_size - 2:
        words_freq = words_freq[:vocab_size - 2]
    for (word, freq) in words_freq:
        word2id[word] = len(word2id)

    if save_to_path:
        with open(save_file, 'w', encoding='utf-8') as f:
            for w in word2id:
                f.write(w + '\t')
                f.write(str(word2id[w]))
                f.write('\n')

    return word2id


def build_word2vec(fname, word2id, save_to_path=None):
    """
    :param fname: 预训练的word2vec.
    :param word2id: 语料文本中包含的词汇集.
    :param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
    :return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
    """
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    return word_vecs


def load_corpus(path, word2id, max_sen_len=128):
    """
    :param path: 样本语料库的文件
    :return: 文本内容contents，以及分类标签labels(onehot形式)
    """
    _, cat2id = cat_to_id()
    contents, labels = [], []
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip('"').split()
            if len(sp) == 0:
                continue
            label = sp[0]
            content = [word2id.get(w, 1) for w in sp[1:]]
            content = content[:max_sen_len]
            if len(content) < max_sen_len:
                content += [word2id['_PAD_']] * (max_sen_len - len(content))
            labels.append(label)
            contents.append(content)
    counter = Counter(labels)
    print('Total sample num：%d' % (len(labels)))
    print('class num：')
    for w in counter:
        print(w, counter[w])

    contents = np.asarray(contents)
    labels = np.array([cat2id[l] for l in labels])

    return contents, labels


def print_hyper_params(model_name):
    num_line = len(open('./modeling/config/'+model_name+'.py','r',encoding = "utf-8").readlines())
    with open('./modeling/config/'+model_name+'.py',encoding = "utf-8") as f:
        for i in range(num_line):
            line = f.readline().rstrip('\n')
            if i >= 5:
                print(line)
        f.close()

