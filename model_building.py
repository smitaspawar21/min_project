import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('./data/processed/train_processed.csv')

# train_data = pd.read_csv('/home/aditya/DSMP2/MLOPS/1PIPELINE-DVC/data/processed/test_processed.csv')


x_train = train_data.iloc[:,0:-1].values

y_train = train_data.iloc[:,-1].values

import yaml

params = yaml.safe_load(open('params.yaml', 'r'))['model_building']

rf = RandomForestClassifier(random_state=params['random_state'],
                             n_estimators = params['n_estimators'])
rf.fit(x_train, y_train)


# save the model to give to next stage

import pickle

pickle.dump(rf, open('model.pkl', 'wb'))  # wb->binary file
