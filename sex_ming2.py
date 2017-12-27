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