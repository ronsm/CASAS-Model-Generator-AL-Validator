# CASAS Model Generator / AL Tools

This repository provides a mechanism to train and validate a handful of machine learning models on CASAS smart home Human Activity Recognition (HAR) data from a committee of models. It re-uses code from '[deep-casas](https://github.com/danielelic/deep-casas)' by Liciotti et al. to pre-process the dataset into a condensed representation:

```
D. Liciotti, M. Bernardini, L. Romeo, and E. Frontoni, ‘A sequential deep learning
application for recognising human activities in smart homes’, Neurocomputing, vol. 396,
pp. 501–513, Jul. 2020, doi: 10.1016/j.neucom.2018.10.104.
```

Any models included in the scikit-learn library can then be applied to the data.

This software forms part of a system to enable simulation of a real smart home with multiple HAR classifiers, to enable experiments in active learning.

## Usage

The ```data.py``` file is taken from [deep-casas](https://github.com/danielelic/deep-casas). Running ```data.py``` generates a numpy-compatible dataset. Unlike in the original code, this version splits the dataset into seperate train/test/validation files at a 50/25/25 split.

The ```train.py``` file implements training of specified models (at upload, these are Random Forest, Decision Tree, and Gradient Boosting Classifier), and test methods to generate predictions from those models on validation data in either simulated real-time or in batches. Example usage is as below.

### Training

This software is for verifying active learning experiments, and so you can train the models with and without additional annotated samples. The train only on the original training set, there are no steps you need to take, but if you wish to include annotations, you must provide the ```annotations.csv``` file from  [HAR Query Committee](https://github.com/ronsm/HAR-Query-Committee) in the ```CSVs``` directory. When this file is in place, running the training command will automatically incorporate this data into the training file: this will make a permanent change to the CSV file on disk.

To train the three selected models, use the following command:
```
python3 train.py train
```

This command will generate first a set of CSV files, which will appear in the ```CSVs``` directory, for training, testing, and validation. The script will then use the training data to train the models.

### Validation

To validate the models with the validation files created by the training step, run either of the commands below, one which will simulate one sample/sec (i.e. real-time) and the other which will batch process all samples.
```
python3 train.py val_real_time
python3 train.py batch
```

### Dataset Generation Only

You can also generate only a dataset split into train/test/validation sets, without training the models. This should only be used to validate that the annotations are being correctly added to the training set, since running the training step will overwrite these files with a potentially different split (if ```shuffle=True``` is enabled in ```train_test_split```.

Use the following command to generate a dataset, without training:
```
python3 train.py dataset_only
```