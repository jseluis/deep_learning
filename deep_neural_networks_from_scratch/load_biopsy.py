import numpy as np
import pandas as pd

def load_biopsy():
    # import data
    biopsy = pd.read_csv('biopsy.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()
    
    # Split in training and test data
    trainI = np.random.choice(biopsy.shape[0], size=300, replace=False)    
    trainIndex=biopsy.index.isin(trainI)    
    train=biopsy.iloc[trainIndex] # training set
    test=biopsy.iloc[~trainIndex] # test set    
    
    # Extract relevant data features
    X_train = train[['V1','V2','V3','V4','V5','V6','V7','V8','V9']].values
    X_test = test[['V1','V2','V3','V4','V5','V6','V7','V8','V9']].values    
    Y_train=(train['class']=='malignant').astype(int).values.reshape((-1,1))
    Y_test=(test['class']=='malignant').astype(int).values.reshape((-1,1))
    
    return X_train, Y_train, X_test, Y_test
