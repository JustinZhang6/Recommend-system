# Recommend-system
   实现了一系列常见的推荐算法，包含“切分训练集与测试集-训练模型-推荐-评估”一整套流程。

# 实现算法
   User Based Collaborative Filtering（基于User的协同过滤）<br>
   Item Based Collaborative Filtering（基于Item的协同过滤）<br>
   基于内容的推荐算法 （进行中）<br>
   矩阵分解（进行中）<br>
   
# 数据集
   Movielens 1M数据集[ml-1m.zip](http://files.grouplens.org/datasets/movielens/ml-1m.zip)
   
# 评价指标：
   指标一：Precision、Recall、Coverage、Popularity<br>
   指标二：MSE、RMSE<br>

# 运行

1. 下载数据集<br>

   下载，并解压到项目Recommend-system文件夹下

2. 运行代码<br>
   eg：<br>
   python user_cf.py

# 注意事项
电影推荐结果并未保留，如果需要此部分数据可自行修改代码。

UserCF算法中，由于用户数量多，生成的相似性矩阵也大，会占用比较多的内存，不过一般电脑都没问题。

ItemCF算法中，每次推荐都需要找出一个用户的所有电影，再为每一部电影找出最相似的电影，运算量比UserCF大，因此推荐的过程比较慢。

# 算法思想
## 基于协同过滤的方法
### UserCF
   >基本思想：相似的用户可能喜欢相似的产品，根据其它用户看的电影+用户相似度，来预测和推荐产品。<br>
   >核心：计算用户之间的相似度<br>
   >相似度计算方法：<br>
   >>简单法：<br>
   >>>对用户a和b，N为数量：<br>
   
      Sim（a,b) = N(a和b共同看过的电影数量) / math.sqrt(N(a单独看过的电影数量) * N（b单独看过的电影数量）)
   
   >>皮尔森系数：  <br>
   >>> ![userCF-sim1](https://github.com/JustinZhang6/Recommend-system/blob/master/image/userCF-sim1.jpg)<br>
   >>![userCF-prid1](https://github.com/JustinZhang6/Recommend-system/blob/master/image/userCF-prid1.jpg)<br>
   
### ItemCF
   >基本思想：相似的产品可能有相似的评价，根据当前用户已评价过的产品+产品相似度，可预测和推荐产品。<br>
   核心：计算产品之间的相似度<br>
   >相似度计算方法：<br>
   >>   对产品x和y，N为数量：<br>
   
      Sim（x,y) = N(产品x和y同时被一个人用过的次数) / math.sqrt(N(x单独被用过的次数)* N(y单独被用过的次数))
   表示如图：<br>
         ![itemCF-sim+calu](https://github.com/JustinZhang6/Recommend-system/blob/master/image/itemCF-sim+calu.jpg)
    
### 两种协同过滤存在的问题
* 都是基于评分矩阵打分，即用户需要有打分如5分，7分，数据获取难。
* 用户打分准确度
* 矩阵稀疏严重
* 冷启动问题严重（新用户/产品的推荐）<br>
不适用于大多真实场景，比如实际有大量用户与产品，难以找最近邻居。
### 改进思路
* 隐式打分：点击、网页浏览、浏览时间、文件下载情况<br>

用户冷启动问题：
* 激励用户主动评分选择倾向
* 基于用户个性<br>

产品冷启动问题：
* 基于内容的方法
* 基于专家的方法<br>

## 基于模型的方法
往往需要很大的算力，在线下预处理，在预测的时候可以直接基于模型预测。<br>
常见的有：<br>
* 基于矩阵分解MF的方法：SVD、PCA。也有一些经典变形：如PMF/SVD++/NMF/FunkSVD/BiasSVD/timeSVD/ConvMF
* 基于关联规则挖掘的方法
* 基于概率的方法：贝叶斯网络。思路是给定打分矩阵，根据如贝叶斯理论，算出可能喜欢的概率。也有复杂的模型如聚类模型、pLSA主题模型等。
* 基于机器学习的方法
* 基于深度学习的方法<br>

# 参考资料
   相关课件<br>
   https://github.com/Lockvictor/MovieLens-RecSys
