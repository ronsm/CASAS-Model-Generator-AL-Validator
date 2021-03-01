# CASAS RNN Model Generator

This repository provides a mechanism to train a handful of RNNs on streaming CASAS smart home Human Activity Recognition (HAR) data from a committee of models. It re-uses code from '[deep-casas](https://github.com/danielelic/deep-casas)' by Liciotti et al. to generate several LSTM-based classifiers, citation below:

```
D. Liciotti, M. Bernardini, L. Romeo, and E. Frontoni, ‘A sequential deep learning
application for recognising human activities in smart homes’, Neurocomputing, vol. 396,
pp. 501–513, Jul. 2020, doi: 10.1016/j.neucom.2018.10.104.
```

This software forms part of a system to enable simulation of a real smart home with multiple HAR classifiers, to enable experiments in active learning.

## Usage

The ```data.py``` and ```models.py``` files are taken from [deep-casas](https://github.com/danielelic/deep-casas). Running ```data.py``` generates a numpy-compatible dataset. Unlike in the original code, this version splits the dataset into seperate train/test/validation files at a 50/25/25 split. The ```models.py``` file describes the models, which contains both the existing (from Liciotti et al.) and newly added models.

The ```train.py``` file implements training of specified models (LSTM, biLSTM, CascadeLSTM), and test methods to generate parallel predictions from those models on test data in simulated real-time. Example usage is as below.

Train:
```
python3 train.py train
```

Test:
```
python3 train.py test
```