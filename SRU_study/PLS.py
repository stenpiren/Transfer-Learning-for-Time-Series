import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import os 


df = pd.read_csv("../datasets/SRU_data.txt", skiprows=1, header=None,sep="  ", dtype="float64",
                 names=["u1","u2","u3","u4", "u5", "y1", "y2"] )

X = df.iloc[:,:5].values
y1 = df.iloc[:,5].values
y2 = df.iloc[:,6].values

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.4)
X_valid, X_test, y1_valid, y1_test, y2_valid, y2_test = train_test_split(X_test, y1_test, y2_test, test_size=0.5) 

model = GridSearchCV(PLSRegression(),
        param_grid={"n_components":[i+1 for i in range(X_train.shape[1])]} ,
        iid = False) 

# 学習
t1 = time()
model.fit(X_train, y1_train)
t2 = time()
model.predict(X_train,y2_train)

condition = "result"
if not os.path.exists("./{}/LSTM".format(condition)):
    os.makedirs("{}/LSTM".format(condition))



# プロット
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.savefig('{}/PLS/plot.png'.format(condition))

# テストデータの予測
predict = model.predict(X_test_time).reshape(-1)
plt.figure(figsize=(18,5))
plt.plot([ i for i in range(1, 1+len(predict))], predict, 'r',label="predicted")
plt.plot([ i for i in range(1, 1+len(predict))], y1_test_time, 'b',label="measured", lw=1, alpha=0.3)
plt.legend(loc="best")
plt.savefig('{}/PLS/test.png'.format(condition))

with open("{}/PLS/result_info.txt".format(condition),"w") as f:
    f.write('TEST_MSE : {:.8f}\n'.format(mse(predict, y1_test_time)))
    f.write('Execute Time : {:.3f}s'.format(t2-t1))