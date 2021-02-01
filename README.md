# CASAS Streaming Data Classifier

This repository provides a mechanism to generate predictions on streaming CASAS smart home Human Activity Recognition (HAR) data from a committee of models. It re-uses code from '[deep-casas](https://github.com/danielelic/deep-casas)' by Liciotti et al. to generate several LSTM-based classifiers, citation below:

```
D. Liciotti, M. Bernardini, L. Romeo, and E. Frontoni, ‘A sequential deep learning
application for recognising human activities in smart homes’, Neurocomputing, vol. 396,
pp. 501–513, Jul. 2020, doi: 10.1016/j.neucom.2018.10.104.
```

The purpose of this software is to simulate a real smart home with multiple HAR classifiers, to enable experiments in active learning.

## Usage

The ```data.py``` and ```models.py``` files are taken from [deep-casas](https://github.com/danielelic/deep-casas). Running ```train.py``` generates a numpy-compatible dataset. Unlike in the original code, this version splits the dataset into seperate train/test files at a 70/30 split. The ```models.py``` file describes the models, which contains both the existing (from Liciotti et al.) and newly added models.

The ```learners_predict.py``` file implements training of specified models (LSTM, biLSTM, CascadeLSTM) and methods to generate parallel predictions from those models on test data in simulated real-time.