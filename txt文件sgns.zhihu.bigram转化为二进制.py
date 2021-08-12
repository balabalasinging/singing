f = open('D:/情感分析5/dataset/sgns.zhihu.bigram', 'r', encoding="UTF-8").read()

#定义转码函数
def encode(s):
    tmp = []
    for c in s:
        tmp.append(bin(ord(c)).replace('0b', ''))
    str_bin = ' '.join(tmp)
    return(str_bin)

b = encode(f)

#存文档
f2 = open('D:/情感分析5/dataset/word2vec.txt','w')
f2.write(b)

#由于在加载预训练模型时，需要用到二进制形式.bin的预训练词向量，
#但是sgns.zhihu.bigram文件是txt文件，所以需要进行转码，上述代码是对该文件进行转码
#然而转化为二进制的时候出现了一个MemoryError报错，找到了原因MemoryError：涉及到了读取、保存、写入，内存不够用了
#由于时间关系，我就自行去找了关于word2vec二分类的二进制预训练词向量模型’wiki_word2vec_50.bin‘来进行训练。
#后期再解决MemoryError对转码的问题。


