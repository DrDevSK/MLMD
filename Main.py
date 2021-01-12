import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, log_loss, auc, f1_score, roc_auc_score, classification_report, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler, binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Loading Data
path='***/Abdominal_Ultrasound_MLMD/'
name='Abdo_US'
train ='***/Abdominal_Ultrasound_MLMD/train_dat.pkl'#55%
val = '***/Abdominal_Ultrasound_MLMD/val_dat.pkl' #15%
test ='***/Abdominal_Ultrasound_MLMD/test_dat.pkl'#30%

#Label (Can change this to train the model using lab tests, combined label, or differential diagnoses label which is used in the study)
Label='PrimaryDx_Label' #From preprocessing file - this is the set of differential diagnoses used to act as the classification label

#Confusion Matrix Plot for Logistic Regression
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path + name + '_Logistic_Regression_CM')
    plt.close()

#Confusion Matrix Plot for Random Forest
def plot_confusion_matrix_RF(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path + name + '_Random_Forest_CM')
    plt.close()

#Confusion Matrix Plot for Neural Network
def plot_confusion_matrix_NN(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path + name + '_Neural_Network_CM')
    plt.close()

#Function for upsampling of the minority class for training data only
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

    #Neural Network Model
    model = keras.Sequential()
    model.add(keras.layers.Dense(2048, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
    model.add(keras.layers.Dense(210, activation=tf.nn.relu))
    model.add(keras.layers.Dense(120, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    #Class weighting parameters
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

    orig_stdout = sys.stdout
    f = open(path + name + '_output_report.txt', 'w')
    sys.stdout = f

    y_pred_NN = model.predict_classes(test_x) #NN model making predictions on the held out test set
    df_predict_out = pd.DataFrame(data=y_pred_NN)
    df_predict_out.to_csv(path+name+'_dx_modelNN_predictions.csv')

    y_pred_NN_proba = model.predict_proba(test_x)
    df_predict_out_proba = pd.DataFrame(data=y_pred_NN_proba)
    df_predict_out_proba.to_csv(path + name+'_dx_modelNN_predictions_NN_proba.csv') #Saving the NN prediction probabilities in a CSV file

    # Compute initial outcome metrics for Neural Network Model.
    cnf_matrix_NN = confusion_matrix(test_y, y_pred_NN)
    np.set_printoptions(precision=2)
    plot_confusion_matrix_NN(cnf_matrix_NN, classes=['No'+name, name], title=name+'NN Confusion matrix')

    tn_NN, fp_NN, fn_NN, tp_NN = confusion_matrix(test_y, y_pred_NN).ravel()
    print("NN True Negatives: ", tn_NN)
    print("NN False Positives: ", fp_NN)
    print("NN False Negatives: ", fn_NN)
    print("NN True Positives: ", tp_NN)

    Precision_NN = tp_NN / (tp_NN + fp_NN)
    print("NN Precision {:0.2f}".format(Precision_NN))

    Recall_NN = tp_NN / (tp_NN + fn_NN)
    print("NN Recall {:0.2f}".format(Recall_NN))

    f1_NN = (2 * Precision_NN * Recall_NN) / (Precision_NN + Recall_NN)
    print("NN F1 Score {:0.2f}".format(f1_NN))

    Specificity_NN = tn_NN / (tn_NN + fp_NN)
    print("NN Specificity {:0.2f}".format(Specificity_NN))

    # Precision-Recall Curve for Neural Network Model
    NN_probs = model.predict_proba(test_x)
    yhat_NN = y_pred_NN
    NN_precision, NN_recall, _ = precision_recall_curve(test_y, NN_probs)
    NN_f1, NN_auc = f1_score(test_y, yhat_NN), auc(NN_recall, NN_precision)
    print('Logistic: f1=%.3f auc=%.3f' % (NN_f1, NN_auc))
    # plot the precision-recall curves
    no_skill = len(test_y[test_y == 1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(NN_recall, NN_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('NN Recall')
    plt.ylabel('NN Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(path + name + '_Neural_Network_AUPRC.png')
    plt.close()

    """The Random Forest Model"""
    rf = RandomForestClassifier(max_depth=30, n_estimators=100,bootstrap = True, max_features = 'sqrt', n_jobs=-1, verbose=1)
    rf.fit(train_x, train_y)

    y_pred_RF2 = (rf.predict_proba(test_x)[:, 1])
    y_pred_RF = (rf.predict_proba(test_x)[:, 1]).astype(bool)
    df_predict_out_RF = pd.DataFrame(data=y_pred_RF2)
    df_predict_out_RF.to_csv(path+name+'_dx_modelNN_predictions_RF_proba.csv') #Saving RF model test set prediction probabilities

    #Initial outcome metrics for the RF model
    cnf_matrix_RF = confusion_matrix(test_y, y_pred_RF)
    np.set_printoptions(precision=2)
    plot_confusion_matrix_RF(cnf_matrix_RF, classes=['No'+name, name], title=name+'Random Forest Confusion matrix')

    tn_RF, fp_RF, fn_RF, tp_RF = confusion_matrix(test_y, y_pred_RF).ravel()
    print("RF True Negatives: ", tn_RF)
    print("RF False Positives: ", fp_RF)
    print("RF False Negatives: ", fn_RF)
    print("RF True Positives: ", tp_RF)

    Precision_RF = tp_RF / (tp_RF + fp_RF)
    print("RF Precision {:0.2f}".format(Precision_RF))

    Recall_RF = tp_RF / (tp_RF + fn_RF)
    print("RF Recall {:0.2f}".format(Recall_RF))

    f1_RF = (2 * Precision_RF * Recall_RF) / (Precision_RF + Recall_RF)
    print("RF F1 Score {:0.2f}".format(f1_RF))

    Specificity_RF = tn_RF / (tn_RF + fp_RF)
    print("RF Specificity {:0.2f}".format(Specificity_RF))

    rf_probs = rf.predict_proba(test_x)
    # keep probabilities for the positive outcome only
    rf_probs = rf_probs[:, 1]
    # predict class values
    yhat_RF = rf.predict(test_x)
    rf_precision, rf_recall, _ = precision_recall_curve(test_y, rf_probs)
    rf_f1, rf_auc = f1_score(test_y, yhat_RF), auc(rf_recall, rf_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (rf_f1, rf_auc))
    # plot the precision-recall curves
    no_skill = len(test_y[test_y == 1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(rf_recall, rf_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('RF Recall')
    plt.ylabel('RF Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(path + name + '_Random_Forest_AUPRC.png')
    plt.close()

    # Extract feature importances for the Random Forest
    fi = pd.DataFrame({'feature': feature,
                       'importance': rf.feature_importances_}). \
        sort_values('importance', ascending=False)

    print("Feature Importance Head")
    print(fi.head(20))

    print("Feature Importance Tail")
    print(fi.tail(20))


    """"The Logistic Regression Model"""
    lr = LogisticRegression(verbose=1, n_jobs=-1)
    lr.fit(train_x,train_y)

    y_pred_LR2 = (lr.predict_proba(test_x)[:, 1])
    y_pred_LR = (rf.predict_proba(test_x)[:, 1]).astype(bool)
    df_predict_out_LR = pd.DataFrame(data=y_pred_LR2)
    df_predict_out_LR.to_csv(path+name+'_dx_modelNN_predictions_LR_proba.csv')

    #Initial outcome metrics for the LR model
    cnf_matrix = confusion_matrix(test_y, y_pred_LR)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(cnf_matrix, classes=['No'+name, name], title=name+' Logistic Regression Confusion matrix')

    tn, fp, fn, tp = confusion_matrix(test_y, y_pred_LR).ravel()
    print("True Negatives: ", tn)
    print("False Positives: ", fp)
    print("False Negatives: ", fn)
    print("True Positives: ", tp)

    Precision = tp / (tp + fp)
    print("Precision {:0.2f}".format(Precision))

    Recall = tp / (tp + fn)
    print("Recall {:0.2f}".format(Recall))

    f1 = (2 * Precision * Recall) / (Precision + Recall)
    print("F1 Score {:0.2f}".format(f1))

    Specificity = tn / (tn + fp)
    print("Specificity {:0.2f}".format(Specificity))

    # predict probabilities
    lr_probs = lr.predict_proba(test_x)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # predict class values
    yhat = lr.predict(test_x)
    lr_precision, lr_recall, _ = precision_recall_curve(test_y, lr_probs)
    lr_f1, lr_auc = f1_score(test_y, yhat), auc(lr_recall, lr_precision)
    # summarize scores
    print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(test_y[test_y == 1]) / len(test_y)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(path + name + 'Logistic_Regression_AUPRC.png')
    plt.close()

    # Create Accuracy and Loss Over Time
    history_dict = history.history
    history_dict.keys()
    # dict_keys(['loss', 'acc', 'val_acc', 'val_loss'])
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 8))
    plt.style.use('fivethirtyeight')
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path + 'NNloss.png')

    plt.clf()  # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    plt.figure(figsize=(10, 8))
    plt.style.use('fivethirtyeight')
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(path + 'NNAccuracy.png')

    #AUROC CURVE FOR ALL MODELS

    y_pred_keras = model.predict(test_x).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    y_pred_rf = rf.predict_proba(test_x)[:, 1]
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(test_y, y_pred_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    y_pred_lr = lr.predict_proba(test_x)[:, 1]
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(test_y, y_pred_lr)
    auc_lr = auc(fpr_lr, tpr_lr)
    plt.figure(1)

    plt.plot(fpr_keras, tpr_keras, label='Neural Network (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = {:.3f})'.format(auc_rf))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (area = {:.3f})'.format(auc_lr))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(name+' Orders_Triage ROC Curve')
    plt.legend(loc='best')
    plt.savefig(path + name + 'AUROC.png')
    plt.close()

    #AUROC CURVE FOR ALL MODELS WITH COMBINED LABEL CM_LABEL

    y_pred_keras = model.predict(test_x).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y2, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    y_pred_rf = rf.predict_proba(test_x)[:, 1]
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(test_y2, y_pred_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    y_pred_lr = lr.predict_proba(test_x)[:, 1]
    fpr_lr, tpr_lr, thresholds_lr = roc_curve(test_y2, y_pred_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure(1)

    plt.plot(fpr_keras, tpr_keras, label='Neural Network (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_rf, tpr_rf, label='Random Forest (area = {:.3f})'.format(auc_rf))
    plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (area = {:.3f})'.format(auc_lr))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(name+' Orders_Triage CM_Label ROC Curve')
    plt.legend(loc='best')
    plt.savefig(path + name + 'AUROC_Combined_Label.png')

if __name__ == '__main__':
    main()
