# Finds Feature Logistic Regression
import configparser
import os
from copy import deepcopy
import random
import time
import numpy as np
from array import array
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

# Logistical Regression Prediction Model
class LR:
    # Initialization Path
    def __init__(self):
        # Initialize Paths to Folders (Train, Test, Relevant Schema)
        self.path_folder_train = r"/Users/thatq/Desktop/ML/Work/Regression_Mega_New/Train/"
        self.path_folder_test = r"/Users/thatq/Desktop/ML/Work/Regression_Mega_New/Test/"
        self.path_fs = r"/Users/thatq/Desktop/ML/Work/Testers/schema_1"
        self.ls_fs = [] # ls_fs holds all the column IDs that are relevant to the data
        self.eta = 0.02  #
        self.lamb = 1E-6 # Lambda prevents code from overfilling or going too high

        # Opens Schema
        """with open(self.path_fs, "r", encoding = "utf-8") as fi:
            for line in fi:
                if line.strip():
                    line = line.strip()
                    # Pound sign (#) before the line signifies that the line is useless/detrimental
                    if not line.startswith("#"):
                        # Take the index, 2nd half is the name of the value 
                        self.ls_fs.append(line.split("|")[0])"""
        with open(self.path_fs, "r", encoding = "utf-8") as fi1:
                lineCount = 0
                for line in fi1:
                    if line.strip():
                        line = line.strip()
                        lineCount += 1
                        if not line.startswith("#"):
                            if lineCount < 57:
                                self.ls_fs.append(lineCount)

    @staticmethod
    # Grab files from folder (This is in the case that there are multiple)
    def getFiles(path_folder):
        try:
            ls_files = []
            for path, subdirs, names in os.walk(path_folder):
                for name in names:
                    ls_files.append(os.path.join(path, name))
            return ls_files
        except Exception:
            print (Exception, "get_latest_file:")
            return ""

    """
        Provides the sigmoid graph which looks something like this
                                            ------------------
                                           /
                                          /
                                         /
                                        /
                                       /
                                      /
                                     /
                                    /
                                   /
                  -----------------
    """
    @staticmethod
    def sigmoid(p):
        return 1.0 / (1 + np.exp(-p))

    # Assigns a random weight to the feature, default at 0.0
    @staticmethod
    def randWeight():
        random.seed(10)
        # return (random.random() - 0.5) * 0.05
        return 0.

    # Runs logistical regression display model
    def run_lr(self):
        ls_best_score = []  # [(auc, log_loss)..]
        # For each important feature, iterate
        for position in self.ls_fs:
            # Prints the position
            print ("Position:", position)
            # Append all to ls_test except the current position
            ls_test = []
            for i in self.ls_fs:
                if not i == position:
                    ls_test.append(i)
            # Dictionary to hold appended weights
            dict_fw = {}
            # Loops total of 5 times to iterate to find lowest log loss value
            # Don't base off AUC
            loop = 5
            # Sets default at 100, this will never stay at 100 unless something is wrong
            current_logloss = 100.
            # Default at 0
            current_auc = 0.
            for i in range(0, loop):
                ls_path_file_train = self.getFiles(self.path_folder_train)
                for path_file_train in ls_path_file_train:
                    with open(path_file_train, "r", encoding = "utf-8") as fi:
                        for line in fi:
                            if line.strip():
                                # Checks for faulty lines $$$ splits the 94 - 5 divisor
                                lss = line.strip().split("|")
                                ls_features = lss[:-2]
                                clk = int(lss[-1])
                                if clk > 1:
                                    clk = 1
                                prediction = 0.0
                                # Creates the logistical regression minus the 1 item
                                for p in ls_test:
                                    # Sets to int value
                                    p = int(p)
                                    try: # Removes faulty lines as well
                                        # Gives the value of prediction
                                        feature = ls_features[p]
                                        if feature != "": # If the feature isn't empty
                                            if p not in dict_fw: # If it's not yet in the dictionary
                                                # Add it to the dictionary with a weight
                                                dict_fw[p] = {feature: self.randWeight()}
                                            else:
                                                # Otherwise, set the feature to the randomweight
                                                if feature not in dict_fw[p]:
                                                    dict_fw[p][feature] = self.randWeight()
                                            # Then add value to prediction
                                            prediction += dict_fw[p][feature]
                                    except IndexError:
                                        pass
                                # Fits prediction to sigmoid model
                                prediction = self.sigmoid(prediction)
                                # Reiterates to normalize values
                                for p in ls_test:
                                    p = int(p)
                                    try:
                                        feature = ls_features[p]
                                        if p in dict_fw:
                                            if feature in dict_fw[p]:
                                                # This normalizes the value by multiplying by a value smaller than 1
                                                dict_fw[p][feature] = dict_fw[p][feature] * (1 - self.lamb) - self.eta * (prediction - clk)
                                                # dict_fw[p][feature] -= self.eta * (prediction - clk)
                                    except IndexError:
                                        pass
                # This Part is the same as earlier
                ls_clk = []
                ls_clk_predicted = []
                ls_path_file_test = self.getFiles(self.path_folder_test)
                for path_file_test in ls_path_file_test:
                    with open(path_file_test, "r", encoding = "utf-8") as fi:
                        for line in fi:
                            if line.strip():
                                lss = line.strip().split("|")
                                ls_features = lss[:-2]
                                clk = int(lss[-1])
                                if clk > 1:
                                    clk = 1
                                prediction = 0.0
                                for p in range(0, len(ls_features)):
                                    try:
                                        feature = ls_features[p]
                                        if p in dict_fw:
                                            if feature in dict_fw[p]:
                                                prediction += dict_fw[p][feature]
                                    except IndexError:
                                        pass
                                prediction = self.sigmoid(prediction)
                                # Up until this part
                                ls_clk.append(clk) #Appends click value 
                                ls_clk_predicted.append(prediction) #Appends predict value
                auc = roc_auc_score(ls_clk, ls_clk_predicted) # Shows accuracy
                logloss = log_loss(ls_clk, ls_clk_predicted) # Show log_loss
                print (auc, logloss) #Print individual logloss
                if logloss < current_logloss:
                    current_logloss = logloss
                    current_auc = auc
                else:
                    break
            print ("Result:", position, current_auc, current_logloss) # Print Best LogLoss
            ls_best_score.append((position, current_auc, current_logloss))
        return ""


if __name__ == "__main__":
    lr = LR()
    ls_best_score = lr.run_lr()
    print (ls_best_score)




