#!/user/bin/env python
# -*- coding:utf-8 -*-
'''
@Author :   zzn
@Data   :   2020-04-06
'''
import time
import math
import random
from collections import defaultdict
from operator import itemgetter


class UserBasedCF(object):
    def __init__(self):
        self.trainset = {}
        self.testset = {}
        self.n_sim_user = 20
        self.n_rec_movie = 10
        self.user_simmat = {}
        self.movie_popular = {}
        self.movie_count = 0.0

    @staticmethod
    def loadfile(filepath):
        ''' return a generator by "yield" ,which help to save RAM. '''
        with open(filepath, 'r') as fp:
            for i, line in enumerate(fp):
                yield line.strip()
                # if i % 1000000 == 0:
                #     print('loading %s(%s)' % (filepath, i))
        print('Load successed!')

    def data_process(self, filepath, p):
        '''
        :param filepath: rating data path
        :return: split dataset to train set and test set

        Dataset format:
        {user1:{movie1:v1, movie2:v2, ..., movieN:vN}
         user2:{...}
         ...
        }
        '''
        len_trainset = 0
        len_testset = 0
        for line in self.loadfile(filepath):
            user, movie, rating, _time = line.split('::')
            if random.random() < p:
                self.trainset.setdefault(user, {})
                # eg: 1196 {'1258': 3, '1': 4}
                self.trainset[user][movie] = int(rating)
                len_trainset += 1
            else:
                self.testset.setdefault(user, {})
                self.testset[user][movie] = int(rating)
                len_testset += 1
        print('train set len =', len_trainset)
        print('test set len =', len_testset)
        print('Trainset user count =', len(self.trainset))
        print('Testset user count =', len(self.testset))

    def calculate_user_sim(self):
        count = 0
        movie2users = dict()
        for user, movies in self.trainset.items():  # items 返回键值对
            for m in movies:
                if m not in movie2users:
                    movie2users[m] = set()
                movie2users[m].add(user)  # 电影每被不同的user评价过，加入相应user
                self.movie_popular[m] = self.movie_popular.get(
                    m, 0) + 1  # 电影m 每次被不同人评价过 +1
        self.movie_count = len(movie2users)
        for movie, users in movie2users.items(
        ):  # {movie :{user1, user2, ...}}
            for u in users:
                self.user_simmat.setdefault(u, defaultdict(int))
                for v in users:
                    if u != v:
                        # 如果两个用户同时评价过同一部电影，那么sim +1
                        self.user_simmat[u][v] += 1
            if count % 1000 == 0:
                print('calu user sim ... (%d)' % count)
            count += 1

        cal_sim_count = 0
        for u, related_users in self.user_simmat.items():
            for v, count in related_users.items():
                self.user_simmat[u][v] = count / \
                                         math.sqrt(len(self.trainset[u]) * len(self.trainset[v]))
                cal_sim_count += 1
                if cal_sim_count % 2000000 == 0:
                    print(
                        'calculating user similarity ... (%d)' % cal_sim_count)

    def recommend(self, user):
        ''' Find K similar user and recommend N movies. '''
        K = self.n_sim_user
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]
        for similar_user, similarity in sorted(self.user_simmat[user].items(),
                                               key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    continue
                rank[movie] = rank.get(movie, 0) + similarity
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def prediction(self, user):
        ''' Find K similar user and predict all movie for user. '''
        K = self.n_sim_user
        rank = {}
        watched_movies = self.trainset[user]
        for similar_user, similarity in sorted(self.user_simmat[user].items(),
                                               key=itemgetter(1), reverse=True)[0:K]:
            for movie in self.trainset[similar_user]:
                if movie in watched_movies:
                    continue
                rank[movie] = rank.get(movie, 0) + similarity
        return rank

    def evalute_recommend(self):
        ''' Test recommend by precision, recall, coverage, popularity. '''
        N = self.n_rec_movie
        hit = 0
        rec_count = 0.0
        test_count = 0.0
        all_rec_movies = set()
        popular_sum = 0
        for i, user in enumerate(self.trainset):
            if i % 500 == 0:
                print('Recommended for %d users' % i)
            test_movies = self.testset.get(user, {})
            rec_movies = self.recommend(user)
            for movie, _ in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
                popular_sum += math.log(1 + self.movie_popular[movie])
            rec_count += N
            test_count += len(test_movies)
        precision = hit / rec_count
        recall = hit / test_count
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        popularity = popular_sum / (1.0 * rec_count)
        print('precision=%.4f\trecall=%.4f\tcoverage=%.4f\tpopularity=%.4f' %
              (precision, recall, coverage, popularity))

    def evalute_prediction(self):
        ''' Test prediction by MSE. '''
        MSE = 0.0
        eval_count = 0
        for i, user in enumerate(self.testset):
            if i % 500 == 0:
                print('Prediction for %d users in testset.' % i)
            test_movie_score = self.testset.get(user, {})   # 获得用户user 所有待预测的movie 及其real_score
            rec_movie_score = self.prediction(user) # 预测user 对所有未评分的电影的分值
            for m, real_score in test_movie_score.items():
                temp = rec_movie_score.get(m, 0) - real_score
                eval_count += 1
                MSE += temp ** 2
                if eval_count % 1000 == 0:
                    print('eval_count(%d) user:%s to movie %s, real_score:%f, predict_score:%f, error:%f'%(eval_count,user,m,real_score,rec_movie_score.get(m,0),temp))
        MSE /= eval_count
        print('MSE = %.4f' % MSE)


def main():
    print('*' * 20, 'User-based collaborative filtering algorithm', '*' * 20)
    usercf = UserBasedCF()
    usercf.data_process('./ratings.dat', p=0.8)
    time_s = time.time()
    usercf.calculate_user_sim()
    time_m = time.time()
    # usercf.evalute_recommend()
    time_er = time.time()
    usercf.evalute_prediction()
    time_ep = time.time()
    print('Time spent calculating is:', time_m - time_s)
    # print('Time spent on recommendations:', time_er - time_m)
    print('Time spend predicting is:', time_ep - time_er)


if __name__ == '__main__':
    main()
