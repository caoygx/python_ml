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
#特征数据，前一项表示身高，后一项表示是否有胡子。如: [178,1] 178表示身高，1表示有胡子
feature=[
	[178,1],
	[155,0],
	[177,0],
	[165,0],
	[169,1],
	[160,0]
]

#特征对应的结果集，是男是女
label=[
	'male',
	'female',
	'male',
	'female',
	'male',
	'female'
	]

#创建决策树对象
clf=tree.DecisionTreeClassifier()

#将特征数据和对应结果发给决策树对象，让它去学习
clf=clf.fit(feature,label)

#测试，给决策树一个特征：身高158,没有胡子。 你告诉我下，是男是女？
result = clf.predict([ [158,0] ])
print(result)