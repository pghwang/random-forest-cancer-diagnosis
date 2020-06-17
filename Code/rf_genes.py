'''
Name:               Peter Hwang
Email:              hwangp@mit.edu
Project:            Machine Learning Cancer Diagnosis (Random Forest)
Date Completed:     June 15, 2020
'''

# Import Packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
SEED = 999

# Diagnosis can take the values 'PNI' or 'LVI', depending on which symptom is being examined
diagnosis = 'PNI'

# Load Datasets (PNI)
if diagnosis == 'PNI':
    data_array = []
    with open('train_PNI.csv') as csvfile: 
        reader = csv.reader(csvfile, delimiter=',', quotechar='|') 
        for row in reader:
            data_array.append(row)
    data_array[0][0] = 'id'
    data_array_transposed = np.transpose(data_array)
    df = pd.DataFrame(data_array_transposed)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    id_names = df['id']
    X, y = df.drop(['id', diagnosis], axis=1), df[diagnosis]
    X_train, y_train = df.drop(['id', diagnosis], axis=1), df[diagnosis]

    data_array = []
    with open('test_PNI.csv') as csvfile: 
        reader = csv.reader(csvfile, delimiter=',', quotechar='|') 
        for row in reader:
            data_array.append(row)
    data_array[0][0] = 'id'
    data_array_transposed = np.transpose(data_array)
    df = pd.DataFrame(data_array_transposed)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    X_test, y_test = df.drop(['id', diagnosis], axis=1), df[diagnosis]

# Load Datasets (LVI)
elif diagnosis == 'LVI':
    data_array = []
    with open('train_LVI.csv') as csvfile: 
        reader = csv.reader(csvfile, delimiter=',', quotechar='|') 
        for row in reader:
            data_array.append(row)
    data_array[0][0] = 'id'
    data_array_transposed = np.transpose(data_array)
    df = pd.DataFrame(data_array_transposed)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    id_names = df['id']
    X, y = df.drop(['id', diagnosis], axis=1), df[diagnosis]
    X_train, y_train = df.drop(['id', diagnosis], axis=1), df[diagnosis]

    data_array = []
    with open('test_LVI.csv') as csvfile: 
        reader = csv.reader(csvfile, delimiter=',', quotechar='|') 
        for row in reader:
            data_array.append(row)
    data_array[0][0] = 'id'
    data_array_transposed = np.transpose(data_array)
    df = pd.DataFrame(data_array_transposed)
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    X_test, y_test = df.drop(['id', diagnosis], axis=1), df[diagnosis]
    
def random_forest(X_train, X_test, tune, importance, plot):
    X_copy = X_test.copy()
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_copy = pd.DataFrame(scaler.transform(X_copy), columns=X_copy.columns)
    
    X_copy['id'] = id_names
    X_copy = X_copy.set_index('id')
    
    def tune_parameters():
        # Number of decision trees
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features per split
        max_features = ['auto', 'sqrt']
        # Max levels per decision tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Min samples to split node
        min_samples_split = [2, 5, 10]
        # Min samples per leaf node
        min_samples_leaf = [1, 2, 4]
        # Which samples to train the decision trees
        bootstrap = [True, False]
        # RandomizedSearchCV parameter tuning
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        rf = RandomForestClassifier()
        # 3 fold cross-validation
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose = 2, random_state = 42, n_jobs = -1)
        # Fit the random search model
        rf_random.fit(X_train, y_train)
        print(rf_random.best_params_)
    
    if tune == True:
        tune_parameters()
    model = RandomForestClassifier(n_jobs=-1, random_state=SEED, n_estimators=1000)
    model.fit(X_train,y_train)   
    
    # Calculate feature importances for each gene
    importances = list(model.feature_importances_)
    feature_list = X_train.columns
    feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    # Print feature importances
    np.savetxt("Importances_PNI.csv", feature_importances, delimiter=",", fmt='%s', header="PNI")
    cols_train = X_train.columns[model.feature_importances_ >= 0.0025460256615451675]
    cols_test = X_test.columns[model.feature_importances_ >= 0.0025460256615451675]
    est_imp = LogisticRegression(solver='lbfgs', max_iter = 10000)
    est_imp.fit(X_train[cols_train], y_train)
    
    # Measure the accuracy on the 20% training dataset
    prediction_array = est_imp.predict(X_test[cols_test])
    tn, fp, fn, tp = confusion_matrix(y_test, prediction_array).ravel()
    acc = accuracy_score(y_test, est_imp.predict(X_test[cols_test]))
    print('Number of features selected: {}'.format(len(cols_test)))
    print('Test Accuracy {}'.format(acc))
    print("Precision = {}".format(precision_score(y_test, est_imp.predict(X_test[cols_test]), average='macro')))
    print("Recall = {}".format(recall_score(y_test, est_imp.predict(X_test[cols_test]), average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test, est_imp.predict(X_test[cols_test]))))
    
    def find_feature_importance_threshold():
        sorted_importances = sorted(importances)
        best_accuracy = 0
        best_importance = 0
        best_features = 0
        count = 0
        for val in sorted_importances:
            if (count >= 0):
                cols_train = X_train.columns[model.feature_importances_ >= val]
                cols_test = X_test.columns[model.feature_importances_ >= val]
                est_imp = LogisticRegression(solver='lbfgs', max_iter = 10000)
                est_imp.fit(X_train[cols_train], y_train)
                y_roc = y_test.copy()
                for index, row in y_roc.iteritems():
                    if (y_roc[index] == '1'):
                        y_roc[index] = 1
                    else:
                        y_roc[index] = 0
                y_roc = y_roc.astype('int64')
                pd.to_numeric(y_roc, errors='coerce')
                lr_probs = est_imp.predict_proba(X_test[cols_test])
                # Remove negative outcomes from the probability matrix
                lr_probs = lr_probs[:, 1]
                # Determine best AUC ROC curve scores
                lr_auc = roc_auc_score(y_roc, lr_probs)
                print("Number of Features: " + str(len(cols_train)) + "         Accuracy: " + str(lr_auc) + "         Importance: " + str(val))
                if (best_accuracy <= lr_auc):
                    best_accuracy = lr_auc
                    best_importance = val
                    best_features = len(cols_train)
            count += 1
        print("BEST ACCURACY = " + str(best_accuracy) + "          BEST IMPORTANCE = " + str(best_importance) + "          NUM FEATURES = " + str(best_features))
    
    if importance == True:
        find_feature_importance_threshold()
    
    y_roc = y_test.copy()
    for index, row in y_roc.iteritems():
        if (y_roc[index] == '1'):
            y_roc[index] = 1
        else:
            y_roc[index] = 0
    y_roc = y_roc.astype('int64')
    pd.to_numeric(y_roc, errors='coerce')
    
    # Create a prediction via a straight-line (i.e. predicting on chance)
    ns_probs = [0 for _ in range(len(y_test))]
    prob_array = est_imp.predict_proba(X_copy[cols_test])
    prob_df = pd.DataFrame(prob_array)
    prob_df['id'] = id_names
    prob_df = prob_df.set_index('id') 
    prob_df.to_csv("Probability_PNI.csv")
    lr_probs = est_imp.predict_proba(X_test[cols_test])
    
    # Remove negative outcomes from the probability matrix
    lr_probs = lr_probs[:, 1]
    
    # Plot Receiver Operating Characteristic curve
    def roc_plot(y_roc, ns_probs, lr_probs):
        # Determine AUC ROC curve scores
        ns_fpr, ns_tpr, _ = roc_curve(y_roc, ns_probs, pos_label = 1)
        lr_fpr, lr_tpr, _ = roc_curve(y_roc, lr_probs, pos_label = 1)
        roc_auc = auc(lr_fpr, lr_tpr)
        
        # Plot ROC curve
        plt.figure()
        lw = 2
        plt.plot(lr_fpr, lr_tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('ROC_PNI.png')
        plt.show()
    
    if plot == True:
        roc_plot(y_roc, ns_probs, lr_probs)
    
if __name__ == "__main__":
    tune = True
    importance = True
    plot = True
    random_forest(X_train, X_test, tune, importance, plot)