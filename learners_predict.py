#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
import colorama
from colorama import Fore, Style
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight
from sklearn.model_selection import train_test_split

import data
import models

# fix random seed for reproducibility
seed = 7
units = 64
epochs = 20

class LearnersPredict(object):
    def __init__(self):
        self.startup_msg()
        self.id = 'learners_predict'

        self.dataset_select = "kyoto11"

        if self.dataset_select in data.datasetsNames:
            msg = 'Selected dataset: ' + self.dataset_select
            self.log(msg)
        else:
            msg = 'Invalid dataset selected.'
            self.log(msg)

    def create_train_test_csvs(self):
        self.log('Creating train/test CSVs...')

        X, Y, dictActivities = data.getData(self.dataset_select)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

        x_test = pd.DataFrame(x_test)
        y_test = pd.DataFrame(y_test)

        x_test.to_csv('x_test.csv', index=False, header=False)
        y_test.to_csv('y_test.csv', index=False, header=False)

        return x_train, x_test, y_train, y_test, dictActivities

    def train_model(self, model_select, X, Y, dictActivities):
        msg = 'Using model: ' + model_select
        self.log(msg)

        # X, Y, dictActivities = data.getData(self.dataset_select)

        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)

        cvaccuracy = []
        cvscores = []
        modelname = ''

        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        k = 0
        for train, test in kfold.split(X, Y):
            print('X_train shape:', X[train].shape)
            print('y_train shape:', Y[train].shape)

            print(dictActivities)

            input_dim = len(X[train])
            X_train_input = X[train]
            X_test_input = X[test]

            no_activities = len(dictActivities)

            if model_select == 'LSTM':
                model = models.get_LSTM(
                    input_dim, units, data.max_lenght, no_activities)
            elif model_select == 'biLSTM':
                model = models.get_biLSTM(
                    input_dim, units, data.max_lenght, no_activities)
            elif model_select == 'Ensemble2LSTM':
                model = models.get_Ensemble2LSTM(
                    input_dim, units, data.max_lenght, no_activities)
            elif model_select == 'CascadeEnsembleLSTM':
                model = models.get_CascadeEnsembleLSTM(
                    input_dim, units, data.max_lenght, no_activities)
            elif model_select == 'CascadeLSTM':
                model = models.get_CascadeLSTM(
                    input_dim, units, data.max_lenght, no_activities)
            else:
                self.log('Invalid model.')
                exit(-1)

            model = models.compileModel(model)
            modelname = model.name

            currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
            csv_logger = CSVLogger(
                model.name + '-' + self.dataset_select + '-' + str(currenttime) + '.csv')
            model_checkpoint = ModelCheckpoint(
                model.name + '-' + self.dataset_select + '-' + str(currenttime) + '.h5',
                monitor='val_accuracy',
                save_best_only=True)

            self.log('Begin training...')
            class_weight = compute_class_weight('balanced', classes=np.unique(Y), y=Y)

            model.fit(X_train_input, Y[train], validation_split=0.2, epochs=epochs, batch_size=64, verbose=1,
                      callbacks=[csv_logger, model_checkpoint])

            self.log('Begin testing ...')
            scores = model.evaluate(
                X_test_input, Y[test], batch_size=64, verbose=1)
            print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))

            self.log('Report:')
            target_names = sorted(dictActivities, key=dictActivities.get)

            classes = model.predict_classes(X_test_input, batch_size=64)
            print(classification_report(
                list(Y[test]), classes, target_names=target_names))
            print('Confusion matrix:')
            labels = list(dictActivities.values())
            print(confusion_matrix(list(Y[test]), classes, labels=labels))

            cvaccuracy.append(scores[1] * 100)
            cvscores.append(scores)

            k += 1

        print('{:.2f}% (+/- {:.2f}%)'.format(np.mean(cvaccuracy), np.std(cvaccuracy)))

        currenttime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        csvfile = 'cv-scores-' + modelname + '-' + \
            self.dataset_select + '-' + str(currenttime) + '.csv'

        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            for val in cvscores:
                writer.writerow([",".join(str(el) for el in val)])

    def train_models(self, x_train, y_train, dictActivities):
        self.train_model('LSTM', x_train, y_train, dictActivities)
        self.train_model('biLSTM', x_train, y_train, dictActivities)
        self.train_model('CascadeLSTM', x_train, y_train, dictActivities)

    def startup_msg(self):
        print(Fore.YELLOW + '* * * * * * * * * * * * * * * * * *')
        print()
        print(Style.BRIGHT + 'Untitled Project' + Style.RESET_ALL + Fore.YELLOW)
        print()
        print(' Developer: Ronnie Smith')
        print(' Email:     ronnie.smith@ed.ac.uk')
        print(' GitHub:    @ronsm')
        print()
        print('* * * * * * * * * * * * * * * * * *')

    def log(self, msg):
        tag = '[' + self.id + ']'
        print(Fore.CYAN + tag, Fore.RESET + msg)

if __name__ == '__main__':
    lp = LearnersPredict()
    x_train, x_test, y_train, y_test, dictActivities = lp.create_train_test_csvs()
    lp.train_models(x_train, y_train, dictActivities)