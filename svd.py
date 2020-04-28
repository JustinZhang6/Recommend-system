#!/user/bin/env python
# -*- coding:utf-8 -*-
'''
@Author :   zzn
@Data   :   2020-04-07
@For    :   SVD 算法，使用numpy里面的SVD实现
'''
import numpy as np
from numpy import linalg as la


def ecludSim(inA, inB):
    '''欧式距离'''
    return 1.0 / (1.0 + la.norm(inA - inB))   # 这里的1/(1+距离)表示将相似度的范围放在0与1之间


def pearsSim(inA, inB):
    '''皮尔逊相关系数距离'''
    if len(inA) < 3:
        return 1.0
    # 参数rowvar=0表示对列求相似度，这里的0.5+0.5*corrcoef()是为了将范围归一化放到0和1之间
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    '''余弦相似度'''
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    if denom != 0:
        return 0.5 + 0.5 * (num / denom)  # 将相似度归一到0与1之间
    else:
        return 0.5


class SVD(object):

    def __init__(self, simMeas=cosSim):
        ''' 相似度衡量的方法默认用余弦相似度 '''
        self.simMeas = simMeas

    @staticmethod
    def loadfile(filepath):
        ''' return a generator by "yield" ,which help to save RAM. '''
        with open(filepath, 'r') as fp:
            for i, line in enumerate(fp):
                yield line.strip()
                # if i % 100000 == 0:
                #     break
                #     print('loading %s(%s)' % (filepath, i))
        print('Load successed!')

    def bd_mat(self, filepath, p):
        ''' 加载和划分数据集，格式xx_data_mat[u, i] 表示用户u对产品i的打分，如果没有打分则记为0.（注意区别array的data[u][i]）

         @:param p : 划分比例，一般是8：2
         得到两个二维矩阵，分别是self.train_data_mat, self.test_data_mat，前者没有部分打分信息，后者只含前者缺失的打分信息，用来检验预测的正确性。
         '''
        n_user = 6040 + 1  # 根据数据集设置
        n_item = 3952 + 1
        self.train_data_mat = np.zeros(shape=(n_user, n_item))
        self.test_data_mat = np.zeros(shape=(n_user, n_item))
        len_trainset = 0
        len_testset = 0

        for line in self.loadfile(filepath):
            user, movie, rating, _time = line.split('::')
            user = int(user)
            movie = int(movie)
            if np.random.random() < p:
                self.train_data_mat[user][movie] = rating
                len_trainset += 1
            else:
                self.test_data_mat[user][movie] = rating
                len_testset += 1
        # print(self.train_data_mat)
        # print(self.test_data_mat)
        self.train_data_mat = np.mat(self.train_data_mat)
        self.test_data_mat = np.mat(self.test_data_mat)
        # print(type(self.train_data_mat))
        print('train set len =', len_trainset)
        print('test set len =', len_testset)
        # print(np.nonzero(self.test_data_mat))

    @staticmethod
    def sigmaPct(sigma, percentage):
        '''按照前k个奇异值的平方和占总奇异值的平方和的百分比percentage来确定k的值,
        后续计算SVD时需要将原始矩阵转换到k维空间
        '''
        sigma2 = sigma ** 2  # 对sigma求平方
        sumsgm2 = sum(sigma2)  # 求所有奇异值sigma的平方和
        sumsgm3 = 0  # sumsgm3是前k个奇异值的平方和
        k = 0
        for i in sigma:
            sumsgm3 += i ** 2
            k += 1
            if sumsgm3 >= sumsgm2 * percentage:
                return k

    def bd_kmat(self, percentage):
        ''' 建立K 维空间.

        将原始打分矩阵降为K 维空间（低维），xformedItems[item:] 表示item 在K 维空间上转化后的值。
        :return self.xformedItems
        '''
        u, sigma, vt = la.svd(self.train_data_mat)
        k = self.sigmaPct(sigma, percentage)  # 确定k的值
        sigmaK = np.mat(np.eye(k) * sigma[:k])  # 构建对角矩阵
        self.xformedItems = self.train_data_mat.T * \
            u[:, :k] * sigmaK.I  # 根据k的值将原始数据转换到k维空间(低维)
        print('k-mat build succ!')

    def calu(self, user, item):
        ''' 预测user对item的打分值

        对于用户user，根据其它电影j 的打分情况，通过self.simMeas相似度计算方法，在K 维空间上，计算item与j 的相似度。
        累加（相似度*对j的打分值）/ 累加（相似度）即为用户user对item的预测分值
        '''
        n = np.shape(self.train_data_mat)[1]  # 电影总数
        simTotal = 0.0
        ratSimTotal = 0.0
        for j in range(n):
            userRating = self.train_data_mat[user, j]   # 对所有电影，获取用户u 对电影j 的评分
            if userRating == 0 or j == item:
                continue   # 只选择曾有评分的
            similarity = self.simMeas(
                self.xformedItems[item, :].T, self.xformedItems[j, :].T)  # 计算item 与j 之间的相似度
            simTotal += similarity
            ratSimTotal += similarity * userRating
        if simTotal == 0:
            return 0
        else:
            return ratSimTotal / simTotal  # 得到对物品item的预测评分

    def evalute(self):
        ''' 评估预测效果，评价指标使用MSE. '''
        MSE = 0.0
        # 找到非0 元素，即找到待预测的item 的坐标。
        row, column = np.nonzero(self.test_data_mat)
        # print(row,column,type(row))
        for i in range(len(row)):
            user = row[i]   # 取得相应id 值
            item = column[i]
            r_hat = self.calu(user, item)
            temp = r_hat - self.test_data_mat[user, item]
            if i % 10000 == 0:
                print(
                    '(%d) user：%d to item:%d, predict value:%.4f and error:%.4f' %
                    (i, user, item, r_hat, temp))
            # print(MSE)
            MSE += temp ** 2
        MSE /= len(row)
        print('MSE =', MSE)

    def recommend(self, user, N=5):
        ''' 产生预测评分最高的N 个推荐结果，默认返回5个

         :return 返回前N 个评分值的item，及其预测评分值. 格式[(item1,score1), (item2,score2), ...]
         '''
        unratedItems = np.nonzero(self.train_data_mat[user, :].A == 0)[
            1]  # 建立一个用户未评分item的列表
        if len(unratedItems) == 0:
            return 'you rated everything'  # 如果都已经评过分，则退出
        itemScores = []
        for item in unratedItems:  # 对于每个未评分的item，都计算其预测评分
            r_hat = self.calu(user, item)
            itemScores.append((item, r_hat))
        itemScores = sorted(
            itemScores,
            key=lambda x: x[1],
            reverse=True)  # 按照item的得分进行从大到小排序
        return itemScores[:N]


def main():
    print('*' * 20, 'Recommendation algorithm based on SVD', '*' * 20)
    svd = SVD(simMeas=cosSim)
    svd.bd_mat('./Data_raw/ml-1m/ratings.dat', p=0.8)
    svd.bd_kmat(percentage=0.8)
    svd.evalute()
    # svd.recommend(user=1) # 测试预测


if __name__ == '__main__':
    main()
