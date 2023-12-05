# Qi Zhao

import os
import pandas as pd
from pandas import DataFrame as df
from Bio import SeqIO
import numpy as np
from sktime.transformations.panel.padder import PaddingTransformer
from sklearn.preprocessing import MultiLabelBinarizer,StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.transformations.panel.rocket import MiniRocket
from joblib import Parallel, delayed

class Mtiot():
    
    def __init__(self,
                 path_label_train = "./data/label.csv",
                 path_ab1_train = './data/NDATA',
                 n_jobs= 1):
        self.n_jobs = n_jobs
        self.path_label_train = path_label_train
        self.path_ab1_train   = path_ab1_train
        
        self.x_train, labels = self.path2data(self.path_ab1_train, self.path_label_train)

        self.mlb = MultiLabelBinarizer()
        self.y = self.mlb.fit_transform(labels)

        self.models = [make_pipeline(
                        PaddingTransformer(),
                        rocket_concat(),
                        StandardScaler(),
                        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), 
                                          class_weight = 'balanced'),
                    ) for _ in range(len(self.mlb.classes_))]
        
        #train
        self.models = Parallel(n_jobs = n_jobs)(delayed(train_model)(model, self.x_train, self.y[:,i]) for i, model in enumerate(self.models))

        
    def predict(self, path_ab1_pre):
        
        self.path_ab1_pre   = path_ab1_pre
        
        x = self.path2data(path_ab1_pre)
        
        self.predict_result = Parallel(n_jobs = self.n_jobs)(delayed(predict_model)(model, x) for model in self.models)
        
        return self.predict_result
    
    def result(self, path_ab1_pre, out_path = 'MTIOT_out.txt'):
        result = df(self.predict(path_ab1_pre))
        samples_name = self.samples_name
        mlb_classes = self.mlb.classes_
        print(1)
        with open(out_path, 'a+') as f:
            for i in result:
                f.write(f"{samples_name[i]}:HPV-{',HPV-'.join(mlb_classes[result[i]==1])}\n")
        
    def path2data(self, path_ab1=None,path_label=None, start = 500):
        
        samples_name = sorted(os.listdir(path_ab1))
        ab1s  = [os.path.join(path_ab1, i) for i in samples_name]
        out_x = df(columns=['green','blue','red','black'],index=range(len(ab1s)))
        for idx,ab1 in enumerate(ab1s):
            abi = list(SeqIO.parse(ab1, "abi"))[0].annotations['abif_raw']
            out_x.iloc[idx,0] = pd.Series(abi['DATA10'][:start:-1])
            out_x.iloc[idx,1] = pd.Series(abi['DATA12'][:start:-1])
            out_x.iloc[idx,2] = pd.Series(abi['DATA11'][:start:-1])
            out_x.iloc[idx,3] = pd.Series(abi['DATA9'][:start:-1])       
            
        if path_label==None:
            self.samples_name = samples_name
            return out_x.applymap(lambda x:(x-x.min())/(x.max()-x.min()))
        else:
            labels = pd.read_csv(path_label, header=None).sort_values(by=1)[0].tolist()
            labels = [label.split(',') for label in labels]
            return out_x.applymap(lambda x:(x-x.min())/(x.max()-x.min())), labels
    
def train_model(model, x, y):
    model.fit(x, y)
    return model

def predict_model(model, x):
    return model.predict(x)

class rocket_concat(BaseEstimator, TransformerMixin):
    
    def __init__(self, model = MiniRocket):
        self.model = model
        self.rockets = [self.model() for _ in range(4)]

    def fit(self, X, y = None):
        self.rockets = [self.rockets[i].fit(X.loc[:,[i]]) for i in range(4)]
        return self

    def transform(self, X):
        return np.concatenate([self.rockets[i].transform(X.loc[:,[i]]) for i in range(4)],axis=1)



