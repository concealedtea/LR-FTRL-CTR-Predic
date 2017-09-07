# -*- coding:utf-8 -*-
import ConfigParser
import os
from copy import deepcopy
import random
import time

from numpy import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss


class LR:
    def __init__(self, eta, lamb, d):
        self.ls_fs = []
        self.eta = eta
        self.lamb = lamb
        self.dict_fw = {}
        for i in xrange(d):
            self.dict_fw[i] = {}

    @staticmethod
    def __sigmoid(p):
        return 1. / (1. + exp(-max(min(p, 35.), -35.)))

    @staticmethod
    def __init_weight():
        random.seed(10)
        # return (random.random() - 0.5) * 0.05
        return 0.

    def predict(self, features):
        p = 0.0
        for i in xrange(0, len(features)):
            feature = features[i]
            if feature != "":
                if feature not in self.dict_fw[i]:
                    self.dict_fw[i][feature] = self.__init_weight()
                p += self.dict_fw[i][feature]
        return self.__sigmoid(p)

    def update(self, features, p, y):
        eta = self.eta
        lamb = self.lamb
        for i in xrange(0, len(features)):
            feature = features[i]
            if feature in self.dict_fw[i]:
                self.dict_fw[i][feature] = self.dict_fw[i][feature] * (1 - lamb) - eta * (p - y)


def __get_all_files(path_folder):
    try:
        ls_files = []
        for path, subdirs, names in os.walk(path_folder):
            for name in names:
                ls_files.append(os.path.join(path, name))
        return ls_files
    except Exception, e:
        print Exception, "get_latest_file:", e
        return ""

if __name__ == "__main__":
    path_folder_train = "/home/ztx/data/data_falcon/TestCTR/train/"
    path_folder_test = "/home/ztx/data/data_falcon/TestCTR/test/"
    path_fs = "/home/ztx/data/data_falcon/TestCTR/schema_forward"
    path_folder_fw = "/home/ztx/data/data_falcon/TestCTR/fw/"

    # 读取schema
    ls_feature_positions = []
    count = 0
    with open(path_fs, "r") as fi:
        for line in fi:
            if line.strip():
                line = line.strip()
                if count <= 56:
                    if not line.startswith("#"):
                        ls_feature_positions.append(count)
                    count += 1
    print ls_feature_positions
    d = len(ls_feature_positions)
    eta = 0.01  #
    lamb = 1E-6
    loop = 5

    lr = LR(eta, lamb, d)

    for i in xrange(loop):
        # train
        ls_path_file_train = __get_all_files(path_folder_train)
        for path_file_train in ls_path_file_train:
            with open(path_file_train, "r") as fi:
                for line in fi:
                    if line.strip():
                        line = line .strip()
                        ls = line.split("|")
                        if len(ls) == 59:
                            clk = 1 if int(ls[-1]) >= 1 else 0
                            ls_features = [ls[int(i)] for i in ls_feature_positions]
                            p = lr.predict(ls_features)
                            lr.update(ls_features, p, clk)
        # test
        ls_clk = []
        ls_clk_predicted = []
        ls_path_file_test = __get_all_files(path_folder_test)
        for path_file_test in ls_path_file_test:
            with open(path_file_test, "r") as fi:
                for line in fi:
                    if line.strip():
                        line = line.strip()
                        ls = line.split("|")
                        if len(ls) == 59:
                            clk = 1 if int(ls[-1]) >= 1 else 0
                            ls_features = [ls[int(i)] for i in ls_feature_positions]
                            p = lr.predict(ls_features)
                            ls_clk.append(clk)
                            ls_clk_predicted.append(p)
        auc = roc_auc_score(ls_clk, ls_clk_predicted)
        logloss = log_loss(ls_clk, ls_clk_predicted)
        print auc, logloss









'''
data_set2 all features:
0.866943211852 0.0824974857568
'''




