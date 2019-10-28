import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from time import time

from keras.layers import Dense, Activation, BatchNormalization
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

model = Sequential([
    Dense(50, input_shape=(5,)),
    BatchNormalization(),
    Activation('relu'),
    Dense(50),
    BatchNormalization(),
    Activation('relu'),
    Dense(1),
    Activation('sigmoid'),
])
model.summary()
model.compile(loss='mse', optimizer='adam')
stop = EarlyStopping(patience=3,verbose=1)

# 学習
t1 = time()
history = model.fit(X_train, y1_train, epochs=20, batch_size=70, validation_data=(X_valid, y1_valid), verbose=1, shuffle=False,callbacks=[stop])
t2 = time()

import os 
if not os.path.exists("./result/ANN"):
    os.makedirs("result/ANN")

# プロット
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper right')
plt.savefig('result/ANN/plot.png')

# テストデータの予測
predict = model.predict(X_test).reshape(-1)
plt.figure(figsize=(18,5))
plt.plot([ i for i in range(1, 1+len(predict))], predict, 'r',label="predicted")
plt.plot([ i for i in range(1, 1+len(predict))], y1_test, 'b',label="measured", lw=1, alpha=0.3)
plt.legend(loc="best")
plt.savefig('result/ANN/test.png')

with open("result/ANN/result_info.txt","w") as f:
    f.write('TEST_MSE : {:.8f}\n'.format(mse(predict, y1_test)))
    f.write('Execute Time : {:.3f}s\n'.format(t2-t1))
    f.write(model.summary())