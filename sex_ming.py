# -*- coding: UTF-8 -*-
'''

运行方式

1.先执行下面两个命令，安装依赖库。
pip install sklearn
pip install scipy

2.运行本程序
python sex_height.py

3.这样即使不会编程的人，复制运行代码也能看出效果。so easy!

闲扯下：这个太不智能了，既然都做到了一个命令就能安装了，为何编译器不能自己去安装这些依赖包呢？还机器学习，人工智能，真正的智能之路还很远吧！


这代码是网上抄的，详细说明看原文 http://blog.csdn.net/qq_28444159/article/details/54428940
'''
from sklearn import tree
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer


#特征数据，前一项表示身高，后一项表示是否有胡子。如: [178,1] 178表示身高，1表示有胡子
'''
feature=[
	['婷',-1],
	['强',-1],
	['刚',-1],
	['静',-1],
	['娟',-1],
	['文',-1]
]
'''



feature=[
	['a',-1],
	['b',-1],
	['c',-1],
	['d',-1],
	['e',-1],
	['f',-1]
]

name_dataset = 'name.csv'

#存放姓名
train_x = []

#存放男女，用[0,1][1,0]表示，不知为何
train_y = []
with open(name_dataset, 'r') as f:
	first_line = True
	for line in f:
		if first_line is True:
			first_line = False
			continue
		sample = line.strip().split(',')
		if len(sample) == 2:
			train_x.append(sample[0])
			if sample[1] == '男':
				train_y.append([0, 1])  # 男
			else:
				train_y.append([1, 0])  # 女
 

max_name_length = max([len(name) for name in train_x])
#print("最长名字的字符数: ", max_name_length)

#名字最大字数设为8
max_name_length = 8

print(train_x)
print(train_y)
 
# 词汇表（参看聊天机器人练习）
counter = 0

#存放每个字出现的次数
vocabulary = {}

for name in train_x:
	counter += 1

	#将 李世民 变成['李','世','民']
	tokens = [word for word in name]

	print(tokens)
	print("\n")

	#统计每个字出现次数，如果李字在vocabulary里，则vocabulary[李]=[李]+1,就是李字出现次数加1
	for word in tokens:
		if word in vocabulary:
			vocabulary[word] += 1
		else:
			vocabulary[word] = 1

print(vocabulary)
 
 #按字符出现次数，从大到小排序. vocabulary.get没搞明白，看样子是获取元素的值
vocabulary_list = [' '] + sorted(vocabulary, key=vocabulary.get, reverse=True)
print('vocabulary_list:')
print(vocabulary_list)

#print(len(vocabulary_list))
 
# 字符串转为向量形式
vocab = dict([(x, y) for (y, x) in enumerate(vocabulary_list)])
print('vocab:')
print(vocab)

#一个宽度为8的数组，存放了一个姓名中每个字在看好序的数组中的索引
train_x_vec = []

#遍历名字
for name in train_x:

	#名字中每个字在排序后的数组中的索引
	name_vec = []

	#遍历名字里的每个字,上面也这样遍历过，合并下，速度是不是更快？
	for word in name:
		name_vec.append(vocab.get(word))

	print('name_vec:')
	print(name_vec)


	#不足8个字的，后面索引号全设为0
	while len(name_vec) < max_name_length:
		name_vec.append(0)

	train_x_vec.append(name_vec)

print('train_x_vec:')
print(train_x_vec)
exit()

vec = DictVectorizer()





measurements = [
    
     {'name': '峰强'},
     {'name': '文静'},
     {'name': '文婷'},
     {'name': '花婷'},
     {'name': '刚强'},
     {'name': '花静'}
     
]
feature = vec.fit_transform(measurements).toarray()

print(feature)




#feature=vec.fit_transform(feature).toarray()

#创建决策树对象
clf=tree.DecisionTreeClassifier()

#feature=preprocessing.normalize(feature)
#feature = tree.LabelEncoder().fit_transform(feature)

# labels = list()
    # 特征提取
    #data = vectorizer.fit_transform(load_data(labels))
    #
    #
    #
#vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
#transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
#feature = vectorizer.fit_transform(feature)

#特征对应的结果集，是男是女
label=[
	'male',
	
	'female',
	'female',
	'female',
	'male',
	'female',
	]


#将特征数据和对应结果发给决策树对象，让它去学习
clf=clf.fit(feature,label)

test=[
    
     {'name': '静婷'},
     {'name': '花花'},
     {'name': '峰刚'},
      {'name': '刚强'},
     {'name': '文静'},
     {'name': '静文'}
]

test = vec.fit_transform(test).toarray()
print(test)


#测试，给决策树一个特征：身高158,没有胡子。 你告诉我下，是男是女？
result = clf.predict(test)
print(result)