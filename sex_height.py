# -*- coding: UTF-8 -*-
from sklearn import tree
#特征数据，前一项表示身高，后一项表示是否有胡子。如: [178,1] 178表示身高，1表示有胡子
feature=[[178,1],[155,0],[177,0],[165,0],[169,1],[160,0]]

#特征对应的结果集，是男是女
label=['male','female','male','female','male','female']

#创建决策树对象
clf=tree.DecisionTreeClassifier()

#将特征数据和对应结果发给决策树对象，让它去学习
clf=clf.fit(feature,label)

#测试，给决策树一个特征：身高158,没有胡子。 你告诉我下，是男是女？
clf.predict([[158,0]])