import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time

from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers import GRU
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os 


df = pd.read_csv("../datasets/SRU_data.txt", skiprows=1, header=None,sep="  ", dtype="float64",
                 names=["u1","u2","u3","u4", "u5", "y1", "y2"] )

X = df.iloc[:,:5].values
y1 = df.iloc[:,5].values
y2 = df.iloc[:,6].values

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(X, y1, y2, test_size=0.4, shuffle=False)
X_valid, X_test, y1_valid, y1_test, y2_valid, y2_test = train_test_split(X_test, y1_test, y2_test, test_size=0.5, shuffle=False) 

def generator(X, y1, y2, time_steps=1000):
    
    n_batches = X.shape[0] - time_steps - 1
    X_time = np.zeros((n_batches, time_steps, X.shape[1]))
    y1_time = np.zeros((n_batches, 1))
    y2_time = np.zeros((n_batches, 1))    
    for i in range(n_batches):
        X_time[i] = X[i:(i+time_steps),:]
        y1_time[i] = y1[i+time_steps]
        y2_time[i] = y2[i+time_steps]
    return X_time, y1_time, y2_time

X_train_time, y1_train_time, y2_train_time = generator(X_train, y1_train, y2_train)
X_valid_time, y1_valid_time, y2_valid_time = generator(X_valid, y1_valid, y2_valid)
X_test_time, y1_test_time, y2_test_time = generator(X_test, y1_test, y2_test)

'''GRUを一層のみにする場合、return_sequences=Falseにする'''
stop = EarlyStopping(patience=10,verbose=1)
def create_model():
    model = Sequential()
    model.add(GRU(input_shape=(X_train_time.shape[1], X_train_time.shape[2]), units = 50, return_sequences=True))
    model.add(BatchNormalization())
    
    # model.add(Dropout(0.3))
    model.add(GRU(units = 50, return_sequences=False))
    model.add(BatchNormalization())
    
    # model.add(Dropout(0.3))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


# 学習
t1 = time()
model1 = create_model()
history1 = model1.fit(X_train_time, y1_train_time, epochs=10, batch_size=500, validation_data=(X_valid_time, y1_valid_time), verbose=1, shuffle=False, callbacks=[stop])
model2 = create_model()
history2 = model2.fit(X_train_time, y2_train_time, epochs=10, batch_size=500, validation_data=(X_valid_time, y2_valid_time), verbose=1, shuffle=False, callbacks=[stop]) 
t2 = time()
history = {"1":history1, "2":history2}
condition = "lin_1000s_50u_2n_100epc"

if not os.path.exists("./{}/GRU".format(condition)):
    os.makedirs("{}/GRU".format(condition))


# プロット

def plot(num, history):
    plt.figure()
    plt.plot(history.history['loss'],marker=".")
    plt.plot(history.history['val_loss'],marker=".")
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig('{}/GRU/plot{}.png'.format(condition, num))

for key,value in history.items():
    plot(key, value)
    
# テストデータの予測
predict1 = model1.predict(X_test_time).reshape(-1)
plt.figure(figsize=(18,5))
plt.plot([ i for i in range(1, 1+len(predict1))], predict1, 'r',label="predicted")
plt.plot([ i for i in range(1, 1+len(predict1))], y1_test_time, 'b',label="measured", lw=1, alpha=0.3)
plt.legend(loc="best")
plt.ylim(0,1)
plt.rcParams["font.size"] = 18
plt.tight_layout()
plt.savefig('{}/GRU/test1.png'.format(condition))

predict2 = model2.predict(X_test_time).reshape(-1)
plt.figure(figsize=(18,5))
plt.plot([ i for i in range(1, 1+len(predict1))], predict1, 'r',label="predicted")
plt.plot([ i for i in range(1, 1+len(predict1))], y1_test_time, 'b',label="measured", lw=1, alpha=0.3)
plt.legend(loc="best")
plt.ylim(0,1)
plt.rcParams["font.size"] = 18
plt.tight_layout()
plt.savefig('{}/GRU/test2.png'.format(condition))

with open("{}/GRU/result_info.txt".format(condition),"w") as f:
    f.write('y1_TEST_MSE : {:.8f}\n'.format(mse(predict1, y1_test_time)))
    f.write('y2_TEST_MSE : {:.8f}\n'.format(mse(predict2, y2_test_time)))
    f.write('Execute Time : {:.3f}s'.format(t2-t1))