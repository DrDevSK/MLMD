import sys
import matplotlib
import os
matplotlib.use('Agg')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, log_loss, auc, f1_score
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import itertools
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
import shap
from tensorflow.keras import Sequential

#Loading Data
path='***/Abdominal_Ultrasound_MLMD/'
name='Abdo_US'
train ='***/Abdominal_Ultrasound_MLMD/train_dat.pkl'
val = '***/Abdominal_Ultrasound_MLMD/val_dat.pkl'
test ='***/Abdominal_Ultrasound_MLMD/test_dat.pkl'

#Label (Can change this to train the model using lab tests, combined label)
Label='PrimaryDx_Label' #From preprocessing file - this is the set of differential diagnoses used to act as the classification label
#Upsampling of the minority class for training data only

#Note if you have a trained model already you can load your model rather than training a new model. This code demonstrates how to
#determine SHAP values if you do not have a pretrained model.

def balance_classes(num, train_dat):
    train_dat_0s = train_dat[train_dat[Label] == 0]
    train_dat_1s = train_dat[train_dat[Label] == 1]
    if num == 1:
        # Bring up 1s
        rep_1 = [train_dat_1s for x in range(train_dat_0s.shape[0] // train_dat_1s.shape[0])]
        keep_1s = pd.concat(rep_1, axis=0)
        train_dat = pd.concat([keep_1s, train_dat_0s], axis=0)
    elif num == 0:
        # Reduce 0s
        keep_0s = train_dat_0s.sample(frac=train_dat_1s.shape[0]/train_dat_0s.shape[0])
        train_dat = pd.concat([keep_0s,train_dat_1s],axis=0)
    return train_dat

def main():
    print(tf.__version__)

    #Loading input data - test, val, train data and dropping different labels (differential diagnosis, combined label, lab tests from the dataset)
    test_dat = pd.read_pickle(test)
    test_dat.drop(name, axis=1, inplace=True)
    test_dat.drop('CM_Label', axis=1, inplace=True)
    test_dat.drop('PrimaryDx', axis=1, inplace=True)
    print (test_dat['PrimaryDx_Label'].value_counts())

    val_dat = pd.read_pickle(val)
    val_dat.drop(name, axis=1, inplace=True)
    val_dat.drop('CM_Label', axis=1, inplace=True)
    val_dat.drop('PrimaryDx', axis=1, inplace=True)
    print (val_dat['PrimaryDx_Label'].value_counts())

    train_dat = pd.read_pickle(train)
    train_dat.drop(name, axis=1, inplace=True)
    train_dat.drop('CM_Label', axis=1, inplace=True)
    train_dat.drop('PrimaryDx', axis=1, inplace=True)
    print (train_dat['PrimaryDx_Label'].value_counts())

    train_dat = train_dat.astype('int')
    test_dat = test_dat.astype('int')
    val_dat=val_dat.astype('int')

    train_dat = balance_classes(1, train_dat)  #Calling function to upsample the minority class

    print("Data Loaded")

    #Extract the labels from the dataset
    test_y = np.array(test_dat.pop(Label))
    train_y = np.array(train_dat.pop(Label))
    val_y = np.array(val_dat.pop(Label))

    #Input features x to the models
    test_x = test_dat
    train_x = train_dat
    val_x = val_dat

    #Getting feature names from column headers
    feature = list(train_x.columns)

    #Transform features by scaling each feature to a given range
    sc_X = MinMaxScaler()
    train_x = sc_X.fit_transform(train_x)
    test_x = sc_X.transform(test_x)
    val_x = sc_X.transform(val_x)

    positive_results = 1 - len([i for i in train_y if i == 1])/len(train_y)
    print(positive_results)
    positive_results = 1 - len([i for i in test_y if i == 1])/len(test_y)
    print(positive_results)

    train_x = np.nan_to_num(train_x)
    test_x = np.nan_to_num(test_x)

    #Note a pretrained model can be loaded instead of training a new model here as an option.
    """Neural Network Model"""
    model = keras.Sequential()
    model.add(keras.layers.Dense(2048, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(keras.layers.Dense(210, activation=tf.nn.relu))
    model.add(keras.layers.Dense(120, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    US = 1000
    No_US = 1

    #Optimizer and Loss Function
    opt = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(class_weight={0:US,1:No_US},loss = "binary_crossentropy", optimizer = opt, metrics=['accuracy'],kernel_regularizer=keras.regularizers.l2(0.05)
                  ,bias_regularizer=keras.regularizers.l2(0.01))

    early_stopping_monitor = EarlyStopping(patience=20)
    history = model.fit(train_x,
                        train_y,
                        epochs=1000,
                        batch_size=1000,
                        validation_data=(val_x, val_y),
                        callbacks=[early_stopping_monitor],
                        verbose=1)

    model.save(path + name+ '_NN.h5') #Saving the NN Model

    #Determing the SHAP values and generating SHAP plots
    #https://github.com/slundberg/shap

    background = train_x[np.random.choice(train_x.shape[0], 10, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(train_x)

    summary_plot = shap.summary_plot(shap_values[0], train_x, feature_names=feature, show=False)
    plt.savefig(path + 'NN_Shap_Summary_plot_3.png', bbox_inches='tight', dpi=600)
    plt.close()

    bar_plot = shap.summary_plot(shap_values, train_x, feature_names=feature, show=False)
    plt.savefig(path + 'NN_Shap_Bar_plot_3.png', bbox_inches='tight', dpi=600)
    plt.close()

    D_plot = shap.dependence_plot("Sex_F", shap_values[0], train_x, interaction_index='Age', feature_names=feature)
    plt.savefig(path + 'Gender_NN_D_plot_3.png', bbox_inches='tight', dpi=600)

    Age_plot = shap.dependence_plot("Age", shap_values[0], train_x, interaction_index=None, feature_names=feature)
    plt.savefig(path + 'Age_NN_D_plot_3.png', bbox_inches='tight', dpi=600)

    Pulse_plot = shap.dependence_plot("Pulse", shap_values[0], train_x, interaction_index=None, feature_names=feature)
    plt.savefig(path + 'Pulse_NN_D_plot_3.png', bbox_inches='tight', dpi=600)

    Pulse_Age_plot = shap.dependence_plot("Age", shap_values[0], train_x, interaction_index='Pulse', feature_names=feature)
    plt.savefig(path + 'Pulse_Age_NN_D_plot_3.png', bbox_inches='tight', dpi=600)

    #Selecting individual patients predictions and generating patient specific SHAP values as examples
    data_for_prediction = test_x[9:10, :]
    background = train_x[0:100, :]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data_for_prediction)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                 feature_names=feature)
    shap.save_html(path + "force_plot.html", force_plot)

    data_for_prediction = test_x[22:23, :]
    background = train_x[0:100, :]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data_for_prediction)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                 feature_names=feature)
    shap.save_html(path + "force_plot2.html", force_plot)

    data_for_prediction = test_x[100:101, :]
    background = train_x[0:100, :]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data_for_prediction)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                 feature_names=feature)
    shap.save_html(path + "force_plot3.html", force_plot)

    data_for_prediction = test_x[8:9, :]
    background = train_x[0:100, :]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data_for_prediction)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                 feature_names=feature)
    shap.save_html(path + "force_plot4.html", force_plot)

    data_for_prediction = test_x[68:69, :]
    background = train_x[0:100, :]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data_for_prediction)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                 feature_names=feature)
    shap.save_html(path + "force_plot5.html", force_plot)

    data_for_prediction = test_x[0:100, :]
    background = train_x[0:100, :]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data_for_prediction)
    force_plot = shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction,
                                 feature_names=feature)
    shap.save_html(path + "summary_force_plot3.html", force_plot)

if __name__ == '__main__':
    main()
