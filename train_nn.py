#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
import colorama
from colorama import Fore, Style
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import sys
from time import perf_counter
from time import sleep
import os.path
import pickle

import data

# fix random seed for reproducibility
seed = 7
units = 64
epochs = 5

class TrainNN(object):
    def __init__(self):
        self.startup_msg()
        self.id = 'train'

        self.dataset_select = "milan"

        if self.dataset_select in data.datasetsNames:
            msg = 'Selected dataset: ' + self.dataset_select
            self.log(msg)
        else:
            msg = 'Invalid dataset selected.'
            self.log(msg)

    def create_train_test_csvs(self):
        self.log('Creating train/test CSVs...')

        X, Y, dictActivities = data.getData(self.dataset_select)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False, train_size=0.1, random_state=seed)

        x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, shuffle=False, test_size=0.5, random_state=seed)

        if os.path.isfile('CSVs/annotations.csv'):
            self.log_warn('[WARNING] Annotations file is present. Annotations will be appended to the training set.')
            annotations = pd.read_csv('CSVs/annotations.csv', skiprows=1, header=None)
            y_annotations = annotations.iloc[:,-1:]
            x_annotations = annotations.drop(annotations.columns[-1], axis=1)

            x_train = pd.DataFrame(x_train)
            x_train = pd.concat([x_train, x_annotations])
            x_train = x_train.values

            y_train = pd.DataFrame(y_train)
            y_annotations = y_annotations.rename(columns={2000:0})
            y_train = pd.concat([y_train, y_annotations])
            y_train = y_train.values

        x_test = pd.DataFrame(x_test)
        y_test = pd.DataFrame(y_test)

        x_validation = pd.DataFrame(x_validation)
        y_validation = pd.DataFrame(y_validation)

        x_train_write = pd.DataFrame(x_train)
        y_train_write = pd.DataFrame(y_train)

        x_train_write.to_csv('CSVs/x_train.csv', index=False, header=False)
        y_train_write.to_csv('CSVs/y_train.csv', index=False, header=False)

        x_test.to_csv('CSVs/x_test.csv', index=False, header=False)
        y_test.to_csv('CSVs/y_test.csv', index=False, header=False)
        
        x_validation.to_csv('CSVs/x_validation.csv', index=False, header=False)
        y_validation.to_csv('CSVs/y_validation.csv', index=False, header=False)

        y_train = y_train.astype('int')

        return x_train, x_test, x_validation, y_train, y_test, y_validation, dictActivities

    def save_model(self, model, name):
        self.log('Saving model...')
        save_file = 'models/' + name + '.p'
        pickle.dump(model, open(save_file, "wb"))

    # Training

    def train_model_1(self, X, Y, dictActivities):
        self.log('Training model 1...')
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        np.save('classes.npy', label_encoder.classes_)

        rf = RandomForestClassifier(n_estimators=100)
        rf = rf.fit(X, Y)

        scores = cross_val_score(rf, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        return rf

    def train_model_2(self, X, Y, dictActivities):
        self.log('Training model 2...')
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        np.save('classes.npy', label_encoder.classes_)

        gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        gbc.fit(X, Y)

        scores = cross_val_score(gbc, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        return gbc

    def train_model_3(self, X, Y, dictActivities):
        self.log('Training model 3...')
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        np.save('classes.npy', label_encoder.classes_)

        dt = tree.DecisionTreeClassifier()
        dt = dt.fit(X, Y)

        scores = cross_val_score(dt, X, Y, cv=5)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

        return dt

    def train_models(self, x_train, y_train, dictActivities):
        model_1 = self.train_model_1(x_train, y_train, dictActivities)
        self.save_model(model_1, 'Model1')

        model_2 = self.train_model_2(x_train, y_train, dictActivities)
        self.save_model(model_2, 'Model2')

        model_3 = self.train_model_3(x_train, y_train, dictActivities)
        self.save_model(model_3, 'Model3')

    # Test

    def load_test_data_and_models(self):
        self.log('Loading test data...')
        x_validation = pd.read_csv('CSVs/x_validation.csv', header=None)
        y_validation = pd.read_csv('CSVs/y_validation.csv', header=None)

        self.log('Loading models...')
        model_1 = pickle.load(open('models/Model1.p', 'rb'))
        model_2 = pickle.load(open('models/Model2.p', 'rb'))
        model_3 = pickle.load(open('models/Model3.p', 'rb'))

        return x_validation, y_validation, model_1, model_2, model_3

    def make_batch_predictions(self, x_test, y_test, model_1, model_2, model_3):
        x_test_data = x_test.values
        y_test_data = y_test.values

        x_test_data = np.array(x_test_data)

        model_1_preds = model_1.predict(x_test_data)
        cr_model_1 = classification_report(y_test_data, model_1_preds)
        print(cr_model_1)

        model_2_preds = model_2.predict(x_test_data)
        cr_model_2 = classification_report(y_test_data, model_2_preds)
        print(cr_model_2)

        model_3_preds = model_3.predict(x_test_data)
        cr_model_3 = classification_report(y_test_data, model_3_preds)
        print(cr_model_3)
    
    # Logging

    def startup_msg(self):
        print(Fore.YELLOW + '* * * * * * * * * * * * * * * * * *')
        print()
        print(Style.BRIGHT + 'CASAS RNN Model Generator' + Style.RESET_ALL + Fore.YELLOW)
        print()
        print(' Developer: Ronnie Smith')
        print(' Email:     ronnie.smith@ed.ac.uk')
        print(' GitHub:    @ronsm')
        print()
        print('* * * * * * * * * * * * * * * * * *')

    def log(self, msg):
        tag = '[' + self.id + ']'
        print(Fore.CYAN + tag, Fore.RESET + msg)
    
    def log_warn(self, msg):
        tag = '[' + self.id + ']'
        print(Fore.CYAN + tag, Fore.LIGHTRED_EX + msg, Fore.RESET)

if __name__ == '__main__':
    tnn = TrainNN()

    first_arg = sys.argv[1]

    if first_arg == "train":
        tnn.log('TRAINING MODE')
        x_train, x_test, x_validation, y_train, y_test, y_validation, dictActivities = tnn.create_train_test_csvs()
        tnn.train_models(x_train, y_train, dictActivities)
    elif first_arg == "val_batch":
        tnn.log('PREDICTION MODE')
        x_validation, y_validation, model_LSTM, model_biLSTM, model_CascadeLSTM = tnn.load_test_data_and_models()
        tnn.make_batch_predictions(x_validation, y_validation, model_LSTM, model_biLSTM, model_CascadeLSTM)
    elif first_arg == "dataset_only":
        tnn.log('DATASET GENERATION ONLY MODE')
        tnn.log_warn('[WARNING] This will result in a dataset that does not correspond to any trained models.')
        x_train, x_test, x_validation, y_train, y_test, y_validation, dictActivities = tnn.create_train_test_csvs()
    else:
        tnn.log('Invalid mode.')