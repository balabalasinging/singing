import jieba
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

stop_words = "  0123456789,?!！.．,:;@，。？：；—…&@#~:;、……～＆＠+*-\t\xad\u3000\u2003\xa0\ufeff＃“”‘’〝〞 \"'＂＇´＇'()（）【】《》＜＞﹝﹞" \
             "<>()[]«»‹›［］「」｛｝〖〗『』"
Path = './dataset/online_shopping_10_cats.csv'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#################################
#对数据进行探索性分析：->Data_Analysis
#################################
def Data_Analysis(Path):
	df = pd.read_csv(Path)
	df = df[['cat', 'review']]
	print("数据总量: %d ." % len(df))
	df.sample(10)
	print("在 cat 列中总共有 %d 个空值." % df['cat'].isnull().sum())
	print("在 review 列中总共有 %d 个空值." % df['review'].isnull().sum())
	df[df.isnull().values == True]
	df = df[pd.notnull(df['review'])]
	d = {'cat': df['cat'].value_counts().index, 'count': df['cat'].value_counts()}
	df_cat = pd.DataFrame(data=d).reset_index(drop=True)
	print(df_cat)
	df_cat.plot(x='cat', y='count', kind='bar', legend=False, figsize=(8, 5))
	plt.ylabel('数量', fontsize=18)
	plt.xlabel('类别', fontsize=18)
	plt.show()
Data_Analysis(Path)

##################################
#开始处理数据，划分训练集，测试集，验证集
##################################

# 读取文件
print('[Preprocess]: reading data.')
df = pd.read_csv(Path)
df = df.drop(['cat'], axis=1)
df_new = df.copy(deep=True)

# 去停用词，分词
print('[Preprocess]: cutting words.')
n, _ = df.shape
for i in tqdm(range(n)):
    label, review = df.iloc[i]
    review = str(review)
    for word in stop_words:
        review = review.replace(word, '')
    if review == '':  # 扔掉空行
        df_new = df_new.drop(labels=[i], axis=0)
        continue
    review = " ".join(jieba.lcut(review))
    df_new.loc[i] = [label, review]
print('[Preprocess]: we have {} examples.'.format(df_new.shape[0]))

df_new['cat_review'] = df_new['review'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stop_words]))
df.head()
df.to_csv('./df.csv', sep=',', header=True, index=True)


# 划分训练验证集测试集
df_new = df_new.reindex(np.random.permutation(df_new.index))  # 打乱顺序
total_len = len(df_new)
# 按6:2:2划分验证集测试集
df_train = df_new[0:int(0.6*total_len)]
df_valid = df_new[int(0.6*total_len):int(0.8*total_len)]
df_test = df_new[int(0.8*total_len):]
print('[Preprocess]: train set contains {} examples.'.format(df_train.shape[0]))
print('[Preprocess]: valid set contains {} examples.'.format(df_valid.shape[0]))
print('[Preprocess]: test set contains {} examples.'.format(df_test.shape[0]))

# 保存数据
print("[Preprocess]: writing files.")
df_train.to_csv('./Raw_Data/train.txt', index=False, sep=" ", header=False)
df_valid.to_csv('./Raw_Data/validation.txt', index=False, sep=" ", header=False)
df_test.to_csv('./Raw_Data/test.txt', index=False, sep=" ", header=False)
print("[Preprocess]: data preprocess finished.")


