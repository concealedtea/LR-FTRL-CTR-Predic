# -*- coding:utf-8 -*-
import configparser
import os
from copy import deepcopy
import random
import time

from math import exp, log, sqrt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


class FTRL():
    def __init__(self, alpha, beta, L1, L2, d):
        self.alpha = alpha  # learning rate
        self.beta = beta  # smoothing parameter for adaptive learning rate
        self.L1 = L1  # L1 regularization, larger value means more regularized
        self.L2 = L2  # L2 regularization, larger value means more regularized
        self.d = d  # d 特征维度
        self.n = {}
        self.z = {}
        self.w = {}
        for i in range(d):
            self.n[i] = {}
            self.z[i] = {}
            self.w[i] = {}

    @staticmethod
    def __init_weight():
        random.seed(10)
        # return (random.random() - 0.5) * 0.05
        return 0.

    def predict(self, features):
        prediction = 0.
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        n = self.n
        z = self.z
        w = {}
        for i in range(0, len(features)):
            w[i] = {}
        for i in range(0, len(features)):
            feature = features[i]
            if feature not in z[i]:
                z[i][feature] = self.__init_weight()
                n[i][feature] = self.__init_weight()
            sign = -1. if z[i][feature] < 0 else 1.
            if sign * z[i][feature] <= L1:
                w[i][feature] = 0.
            else:
                w[i][feature] = (sign * L1 - z[i][feature]) / ((beta + sqrt(n[i][feature])) / alpha + L2)
            prediction += w[i][feature]
        self.w = w
        return 1. / (1. + exp(-max(min(prediction, 35.), -35.)))

    def update(self, features, p, y):
        '''
        :param x: feature, a list of indices
        :param p: predictions
        :param y: real clicks
        :return:
        '''

        alpha = self.alpha
        n = self.n
        z = self.z
        w = self.w
        g = p - y

        for i in range(0, len(features)):
            feature = features[i]
            sigma = (sqrt(n[i][feature] + g * g) - sqrt(n[i][feature])) / alpha
            z[i][feature] += g - sigma * w[i][feature]
            n[i][feature] += g * g


def get_all_files(path_folder):
    try:
        ls_files = []
        for path, subdirs, names in os.walk(path_folder):
            for name in names:
                ls_files.append(os.path.join(path, name))
        return ls_files
    except Exception:
        print (Exception, "get_latest_file:")
        return ""

if __name__ == "__main__":
    alpha = 0.01  # learning rate
    beta = 0.1  # smoothing parameter for adaptive learning rate
    L1 = 0.  # L1 regularization, larger value means more regularized
    L2 = 1.  # L2 regularization, larger value means more regularized

    path_folder_train = r"/Users/thatq/Desktop/ML/Work/Regression_Mega_New/Train/"
    path_folder_test = r"/Users/thatq/Desktop/ML/Work/Regression_Mega_New/Test/"
    path_fs = r"/Users/thatq/Desktop/ML/Work/Testers/schema_1"

    ls_feature_positions = []
    with open(path_fs, "r", encoding = "utf-8") as fi1:
        lineCount = 0
        for line in fi1:
            if line.strip():
                line = line.strip()
                lineCount += 1
                if not line.startswith("#"):
                    if lineCount <= 56:
                        ls_feature_positions.append(lineCount)
    d = len(ls_feature_positions)
    ls_best_score = []  # [(auc, log_loss)..]
    loop = 20
    # for i in ls_feature_positions:
    '''ls_test = []
    for h in ls_feature_positions:
        if not h == i:
            ls_test.append(h)
    print("Position:", i)'''
    current_logloss = 100.
    # Default at 0
    current_auc = 0.
    ftrl = FTRL(alpha, beta, L1, L2, d)
    for q in range (0,loop):
        # train
        ls_path_file_train = get_all_files(path_folder_train)
        for path_file_train in ls_path_file_train:
            with open(path_file_train, "r", encoding = "utf-8") as fi:
                for line in fi:
                    if line.strip():
                        lss = line.strip().split("|")
                        ls_features = lss[:-2]
                        clk = int(lss[-1])
                        if clk > 1:
                            clk = 1
                        ls_sub_features = [ls_features[q] for q in ls_feature_positions]
                        prediction = ftrl.predict(ls_sub_features)
                        ftrl.update(ls_sub_features, prediction, clk)
                        #if clk == 1:
                        #    print (prediction)
    
        # test
        ls_clk = []
        ls_clk_predicted = []
        ls_path_file_test = get_all_files(path_folder_test)
        for path_file_test in ls_path_file_test:
            with open(path_file_test, "r", encoding = "utf-8") as fi:
                for line in fi:
                    if line.strip():
                        lss = line.strip().split("|")
                        ls_features = lss[:-2]
                        clk = int(lss[-1])
                        if clk > 1:
                            clk = 1
                        ls_sub_features = [ls_features[q] for q in ls_feature_positions]
                        prediction = ftrl.predict(ls_sub_features)
                        # if clk == 1:
                        #     print prediction, clk
                        ls_clk.append(clk)
                        ls_clk_predicted.append(prediction)
        auc = roc_auc_score(ls_clk, ls_clk_predicted)
        logloss = log_loss(ls_clk, ls_clk_predicted)
        print (auc, logloss)
        if logloss < current_logloss:
            current_logloss = logloss
            current_auc = auc
        else:
            pass
    print ("Result:", current_auc, current_logloss) # Print Best LogLoss
    ls_best_score.append((current_auc, current_logloss))
    """
    Total AUC/LogLoss: 0.559306458895 0.129277765985
    """
    










