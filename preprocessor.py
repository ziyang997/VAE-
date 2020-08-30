"""
Ziwei Zhu
Computer Science and Engineering Department, Texas A&M University
zhuziwei@tamu.edu
"""
import numpy as np
import pandas as pd

class ml1m:
    def __init__(self, args):
       self.batch_size = args.batch_size
       self.train_df = pd.read_csv('./data/ml-1m/train.csv')
       self.test_df = pd.read_csv('./data/ml-1m/test.csv')
       self.num_users = np.max(self.train_df['userId'])
       self.num_items = np.max(self.train_df['movieId'])
       self.train_mat = self.train_df.values
       self.test_mat = self.test_df.values
       self.train_set, self.train_R = self.get_train_set()
       self.test_R = self.get_test_R()

    def get_test_R(self):
        num_users = np.max(self.test_df['userId'])
        num_items = np.max(self.test_df['movieId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix
        for i in range(len(self.test_df)):
            user_idx = int(self.test_mat[i, 0]) - 1
            item_idx = int(self.test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1
        return test_R
    def get_train_set(self):
       train_R = np.zeros((self.num_users, self.num_items))
       cur_user = self.train_mat[0, 0] - 1
       train_set = {}
       train_items = []
       for i in range(len(self.train_df)):
           user_idx = int(self.train_mat[i, 0]) - 1
           item_idx = int(self.train_mat[i, 1]) - 1
           train_R[user_idx, item_idx] = 1
           if cur_user == user_idx:
               train_items.append(item_idx)
           else:
               train_set[cur_user] = train_items
               cur_user = user_idx
               train_items = []
       train_set[cur_user] = train_items
       return train_set, train_R

    def sample (self):
        users = np.random.choice(np.arange(self.num_users), self.batch_size, replace=False)
        pos_items, neg_items = [], []
        for u in users:
            pos_items.append(np.random.choice(self.train_set[u], 1)[0])
            while True:
                neg_id = np.random.randint(low=0, high=self.num_items, size=1)[0]
                if neg_id not in self.train_set[u]:
                    neg_items.append(neg_id)
                    break
        return users, pos_items, neg_items

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/ml-1m/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/ml-1m/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        return train_R, vali_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/ml-1m/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['movieId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/ml-1m/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            #train_R[user_idx, item_idx] = 1

        return train_R, test_R

class Pinterest:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/p/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/p/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        return train_R, vali_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/p/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['movieId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/p/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R




class yelp:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/yelp/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/yelp/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['itemId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        return train_R, vali_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/yelp/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['itemId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/yelp/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['itemId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R

