import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score, precision_score

rf = pickle.load(open('model.pkl', 'rb'))

test_data = pd.read_csv('./data/processed/test_processed.csv')

x_test = test_data.iloc[:,0:-1].values

y_test = test_data.iloc[:,-1].values

# predict on test

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=' Approved')
recall = recall_score(y_test, y_pred, pos_label=' Approved')
# note => here leading space is there for target variable

metrics_dict = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall
}

# create a file to dump this metrics dict

import json

with open('metrics.json', 'w') as f:
    json.dump(metrics_dict, f, indent= 5)