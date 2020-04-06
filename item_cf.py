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


class ItemBasedCF(object):
    def __init__(self):
        self.trainset = {}
        self.testset = {}
        self.n_sim_movie = 20
        self.n_rec_movie = 10
        self.movie_simmat = {}
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

    def calculate_movie_sim(self):
        count = 0
        for user, movies in self.trainset.items():  # items 返回键值对
            for m in movies:
                self.movie_popular[m] = self.movie_popular.get(
                    m, 0) + 1  # 电影m 每次被不同人评价过 +1
                self.movie_simmat.setdefault(m, defaultdict(int))
                for m2 in movies:
                    if m != m2:
                        # 如果两部电影被同一个人评价过，那么sim +1
                        self.movie_simmat[m][m2] += 1

            if count % 1000 == 0:
                print('calu movie sim ... (%d)' % count)
            count += 1
            # print(user,movies)
        self.movie_count = len(self.movie_popular)
        cal_sim_count = 0
        for m1, related_movies in self.movie_simmat.items():
            for m2, count in related_movies.items():
                self.movie_simmat[m1][m2] = count / \
                    math.sqrt(self.movie_popular[m1] * self.movie_popular[m2])
                cal_sim_count += 1
                if cal_sim_count % 2000000 == 0:
                    print(
                        'calculating item similarity ... (%d)' %
                        cal_sim_count)

    def recommend(self, user):
        ''' Find K similar movies and recommend N movies based on user watched. '''
        K = self.n_sim_movie
        N = self.n_rec_movie
        rank = {}
        watched_movies = self.trainset[user]
        for movie, rating in watched_movies.items():
            for related_movie, similarity in sorted(
                    self.movie_simmat[movie].items(), key=itemgetter(1), reverse=True)[:K]:
                if related_movie in watched_movies:
                    continue
                rank[related_movie] = rank.get(
                    related_movie, 0) + similarity % rating
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:N]

    def prediction(self, user):
        ''' Predict all movie for user based on user watched '''
        rank = {}
        watched_movies = self.trainset[user]
        for movie, rating in watched_movies.items():
            for other_movie, similarity in self.movie_simmat[movie].items():
                if other_movie in watched_movies:
                    continue
                rank[other_movie] = rank.get(
                    other_movie, 0) + similarity % rating
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
            test_movie_score = self.testset.get(user, {})
            rec_movie_score = self.prediction(user)
            for m, real_score in test_movie_score.items():
                temp = real_score - rec_movie_score.get(m, 0)
                eval_count += 1
                MSE += temp**2
        MSE /= eval_count
        print('MSE = %.4f' % MSE)


def main():
    print('*'*10,'Item-based collaborative filtering algorithm','*'*10)
    itemcf = ItemBasedCF()
    itemcf.data_process('./ratings.dat', 0.8)
    time_s = time.time()
    itemcf.calculate_movie_sim()
    time_m = time.time()
    itemcf.evalute_recommend()
    time_er = time.time()
    itemcf.evalute_prediction()
    time_ep = time.time()
    print('Time spent calculating is:',time_m - time_s)
    print('Time spent on recommendations:',time_er - time_m)
    print('Time spend predicting is:',time_ep - time_er)

if __name__ == '__main__':
    main()
