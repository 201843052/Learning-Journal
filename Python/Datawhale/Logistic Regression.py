# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 15:18:05 2021

@author: josia
"""
import pandas as pd
import numpy as np
from sklearn import model_selection as ms
import os
os.chdir("H:/360MoveData/Users/josia/Documents/GitHub/Learning-Journal/Python/Datawhale")

print("This script is in Git repo - python - datawhale")

# Coding out logisstic regression using data from page 

df = pd.read_csv("Watermelon.csv")

pred = df[["Concentration","Glucose Rate"]]  # predictors
pred["1"] = 1
resp = df["output"]  # respoonse

#  Training and Testing
X_train, X_test, y_train, y_test = ms.train_test_split(pred, resp, test_size=0.3, random_state=42)


#  y = beta * X
#  beta_hat = (X_t*X)**(-1) *X_t*y

def link(z):
    return 1/(1+np.exp(-z))

def logit(p):
    return np.log(p/(1-p))

def loss(beta,x,y):
    nrow = x.shape
    p = link(x.dot(beta))
    J = (-y).T.dot(np.log(p)) - 