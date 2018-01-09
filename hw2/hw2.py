import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.externals import joblib
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
for name in data.columns:
    data[name] = pd.Categorical(data[name]).codes

x = data[data.columns[:-1]]
y = data[data.columns[-1]]

X_train,X_valid,y_train,y_valid = train_test_split(x,y,test_size=0.3,random_state=0)

model = LogisticRegression(penalty='l2',C=10,class_weight="balanced")
model.fit(X_train,y_train)

y_train_pred = model.predict(X_train)
print("train set accuracy:",accuracy_score(y_train,y_train_pred))
print("train set precision:",precision_score(y_train,y_train_pred))
print("train set recall",recall_score(y_train,y_train_pred))
print("train set f1 score",f1_score(y_train,y_train_pred))


y_valid_pred = model.predict(X_valid)
print("valid set precision:",precision_score(y_valid,y_valid_pred))
print(" valid set precision：",precision_score(y_valid,y_valid_pred))
print("valid set recall：",recall_score(y_valid,y_valid_pred))
print("valid set F1 score:",f1_score(y_valid,y_valid_pred))

y_valid_prob = model.predict_proba(X_valid)
y_valid_prob = y_valid_prob[:,1]

fpr,tpr,thresholds = metrics.roc_curve(y_valid,y_valid_prob)
auc = metrics.auc(fpr,tpr)
print("valid set AUC",auc)

plt.figure(facecolor="w")
plt.plot(fpr,tpr,'r-',lw=2,alpha=0.8,label="AUC%0.3f"%auc)
plt.plot((0,1),(0,1),c = 'b', lw = 1.5,ls = "--",alpha = 0.7)
plt.xlim((-0.01,1.01))
plt.ylim((-0.01,1.01))
plt.xlabel("False Position Rate",fontsize=14)
plt.ylabel("True Position Rate",fontsize=10)
plt.grid(b=True)
plt.legend(loc="lower right",fancybox=True,fontsize = 14)
plt.title("hw2 data ROC curve and AUC sore",fontsize=17)
plt.show()