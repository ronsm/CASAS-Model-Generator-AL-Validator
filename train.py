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
from tensorflow import keras
import sys
from time import perf_counter
from time import sleep

import data
import models

# fix random seed for reproducibility
seed = 7
units = 64
epochs = 5

class LearnersPredict(object):
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

        x_train, x_test, y_train, y_test = train_test_split(X, Y, shuffle=False, test_size=0.5, random_state=seed)

        x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, shuffle=False, test_size=0.5, random_state=seed)

        x_test = pd.DataFrame(x_test)
        y_test = pd.DataFrame(y_test)

        x_validation = pd.DataFrame(x_validation)
        y_validation = pd.DataFrame(y_validation)

        x_train.to_csv('x_train.csv', index=False, header=False)
        y_train.to_csv('x_train.csv', index=False, header=False)

        x_test.to_csv('x_test.csv', index=False, header=False)
        y_test.to_csv('y_test.csv', index=False, header=False)
        
        x_validation.to_csv('x_validation.csv', index=False, header=False)
        y_validation.to_csv('y_validation.csv', index=False, header=False)

        return x_train, x_test, y_train, y_test, dictActivities

    # Training

    def train_model(self, model_select, X, Y, dictActivities):
        msg = 'Using model: ' + model_select
        self.log(msg)

        # X, Y, dictActivities = data.getData(self.dataset_select)

        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        np.save('classes.npy', label_encoder.classes_)

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

    # Test

    def load_test_data_and_models(self):
        self.log('Loading test data...')
        x_test = pd.read_csv('x_test.csv', header=None)
        y_test = pd.read_csv('y_test.csv', header=None)

        self.log('Loading models...')
        model_LSTM = keras.models.load_model('LSTM.h5')
        model_biLSTM = keras.models.load_model('biLSTM.h5')
        model_CascadeLSTM = keras.models.load_model('CascadeLSTM.h5')

        return x_test, y_test, model_LSTM, model_biLSTM, model_CascadeLSTM

    def make_single_prediction(self, model, sample):
        y_pred = model.predict_classes(sample)
        return y_pred

    def make_sequential_predictions(self, x_test, y_test, model_LSTM, model_biLSTM, model_CascadeLSTM):
        x_test_data = x_test.values
        y_test_data = y_test.values

        x_test_data = np.array(x_test_data)

        for i in range(0, len(y_test)):
            if i > 0:
                start_time = perf_counter()

            sample = x_test_data[i]
            # Keras LSTM expects 3D tensor even if batch size is one sample
            sample = np.expand_dims(sample, axis=0)
            print('Sample:', sample)

            y_pred_LSTM = self.make_single_prediction(model_LSTM, sample)
            y_pred_biLSTM = self.make_single_prediction(model_biLSTM, sample)
            y_pred_CascadeLSTM = self.make_single_prediction(model_CascadeLSTM, sample)

            self.log('Prediction analysis:')
            print('Actual:', y_test_data[i], 'Predictions: LSTM =', y_pred_LSTM, ', biLSTM =', y_pred_biLSTM, ', Cascade:LSTM = ', y_pred_CascadeLSTM)

            if i > 0:
                end_time = perf_counter()
                time_taken = end_time - start_time
                delay_time = 1.0 - time_taken

                if delay_time >= 0.0:
                    print('Prediction time was:', time_taken, ', sleeping for:', delay_time, 'seconds')
                    sleep(delay_time)
                else:
                    self.log_warn('Prediction took longer than 1 second! System is not keeping up with real-time.')
    
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
    lp = LearnersPredict()

    first_arg = sys.argv[1]

    if first_arg == "train":
        lp.log('TRAINING MODE')
        x_train, x_test, y_train, y_test, dictActivities = lp.create_train_test_csvs()
        lp.train_models(x_train, y_train, dictActivities)
    elif first_arg == "test":
        lp.log('PREDICTION MODE')
        x_test, y_test, model_LSTM, model_biLSTM, model_CascadeLSTM = lp.load_test_data_and_models()
        lp.make_sequential_predictions(x_test, y_test, model_LSTM, model_biLSTM, model_CascadeLSTM)
    elif first_arg == "dataset_only":
        lp.log('DATASET GENERATION ONLY MODE')
        lp.log_warn('[WARNING] This will result in a dataset that does not correspond to any trained models.')
        x_train, x_test, y_train, y_test, dictActivities = lp.create_train_test_csvs()
    else:
        lp.log('Invalid mode.')