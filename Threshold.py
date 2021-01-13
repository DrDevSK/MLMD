import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

value = 0.10
metric = "tpr"
name="Abdo_US"

file="***/threshold_outputs_"+str(value)+metric+".txt" #Saving output file
sys.stdout = open(file, "w")

print("Threshold Values for ",metric," ",value)

#Function for calculating outcome metrics based on threshold set above
def calculate_threshold(preds, targets, indicator_value, indicator='tpr'):
    sorted_index = np.argsort(preds)[::-1]

    preds = preds[sorted_index]
    targets = targets[sorted_index]

    threshold = 1.0
    closeness = 1.0
    found = False

    for i in range(preds.shape[0]):
        curr_threshold = preds[i]
        preds_i = preds >= curr_threshold

        cm = confusion_matrix(targets, preds_i).astype(float)

        tn = cm[0, 0]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tp = cm[1, 1]
        value = 0
        if indicator == 'tnr':
            value = tn/(tn+fp)
        elif indicator == 'npv':
            if tn+fn == 0:
                value = 0
            else:
                value = tn/(tn+fn)
        elif indicator == 'ppv':
            value = tp/(tp+fp)
        elif indicator == 'tpr':
            value = tp/(tp+fn)

        if indicator == 'ppv' or indicator == 'npv':
            if abs(value-indicator_value) <= closeness:
                closeness = abs(value-indicator_value)
                threshold = curr_threshold
        else:
            if abs(value-indicator_value) < closeness:
                closeness = abs(value-indicator_value)
                threshold = curr_threshold
                found = True
            elif found and closeness < abs(value-indicator_value):
                break

    return threshold

print("______________________________________________________________________________________________________________________________")
print("_______________________________________________________Primary_Dx_Label_Thresholds____________________________________________")
print("______________________________________________________________________________________________________________________________")

# Load data
target_data=pd.read_pickle('***/test_dat.pkl')
targets = target_data['PrimaryDx_Label'].to_numpy()
print (target_data['PrimaryDx_Label'].value_counts())

print("_______________________________________________________NN_____________________________________________________________________")

preds_NN=pd.read_csv('***/NN_proba.csv')
preds = preds_NN["0"].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)



print("_______________________________________________________RF__________________________________________________")


preds_RF=pd.read_csv('***/RF_proba.csv')
preds = preds_RF['0'].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)

print("_______________________________________________________LR__________________________________________________")

preds_LR=pd.read_csv('***/LR_proba.csv')
preds = preds_LR['0'].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)

print("______________________________________________________________________________________________________________________________")
print("_______________________________________________________Combined_Label_Thresholds______________________________________________")
print("______________________________________________________________________________________________________________________________")

# Load data
targets = target_data['CM_Label'].to_numpy()
print(target_data['CM_Label'].value_counts())

print("_______________________________________________________NN_____________________________________________________________________")

preds = preds_NN["0"].to_numpy()
# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)



print("_______________________________________________________RF__________________________________________________")


preds = preds_RF['0'].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)

print("_______________________________________________________LR__________________________________________________")

preds = preds_LR['0'].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)


print("______________________________________________________________________________________________________________________________")
print("_______________________________________________________Test_Label_Thresholds__________________________________________________")
print("______________________________________________________________________________________________________________________________")

# Load data
targets = target_data[name].to_numpy()
print (target_data[name].value_counts())

print("_______________________________________________________NN_____________________________________________________________________")

preds = preds_NN["0"].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)



print("_______________________________________________________RF__________________________________________________")

preds = preds_RF['0'].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)

print("_______________________________________________________LR__________________________________________________")

preds = preds_LR['0'].to_numpy()

# Test the original CM at 0.5 threshold
preds_05 = preds >= 0.5
cm = confusion_matrix(targets, preds_05)
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

cm = confusion_matrix(targets, preds_05)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("Confusion Matrix for Original Set at 0.5: ")
print (cm)
print ("")

# Get the threshold that results in us getting the ideal value we want for a specific indicator
threshold = calculate_threshold(preds, targets, value, metric)

print ("Ideal model threshold: " + str(threshold))
preds_i = preds >= threshold

cm = confusion_matrix(targets, preds_i)
print (cm)

cm = confusion_matrix(targets, preds_i)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print (cm)

sys.stdout.close()
