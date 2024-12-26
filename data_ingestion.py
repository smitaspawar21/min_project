import pandas as pd
import numpy as np

import os

df = pd.read_csv(r"D:\loan_approval_dataset.csv")

df.columns = df.columns.str.replace(' ', '')

x = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status']

import yaml

test_size = yaml.safe_load(open('params.yaml', 'r'))['data_ingestion']['test_size']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    random_state=0, stratify= y,
                                                    test_size=test_size)

train_data = pd.concat([x_train, y_train], axis=1)
test_data = pd.concat([x_test, y_test], axis=1)

# creating a folder for storing this data locally

data_path = os.path.join('data', 'raw')

os.makedirs(data_path)


# save to these created path

train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index= False)