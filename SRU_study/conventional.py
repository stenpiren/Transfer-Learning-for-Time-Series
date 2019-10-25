import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sample_selector import JIT, MW, TD
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import time
import tensorflow as tf 

t1 = time.time()
df = pd.read_csv("../datasets/debutanizer_data.txt", skiprows=1, header=None,sep="  ", dtype="float64",
                 names=["u1","u2","u3","u4", "u5", "u6", "u7", "y"] )

idx_train = df.index[:500] # 初期状態ではデータベースに500個のデータが存在
idx_test = df.index[500:] 
columns = list(df.columns)
columns_y = columns.pop(-1) # columnsから目的変数y2を削除し、そのy2をcolumns_yに代入
columns_x = columns

delay = 8 # 遅れ時間(TDのself.deltaよりも小さい値にしなければならない)
n_window = 40 # 回帰モデルに利用する1バッチ当たりのデータ数

X_train = df.loc[:499, columns_x] 
y_train = df.loc[:(499-delay),[columns_y]] 
X_test = df.loc[500:, columns_x]
y_test = df.loc[(500-delay):,[columns_y]]


samplings = {
    "TD" : TD(n_window),
    "JIT" : JIT(n_window),
    "MW" : MW(n_window)    
}

results = pd.DataFrame()
for name, sampling in samplings.items():
    
        print("{}".format(name))
        results["y_pred_{}_rnn".format(name)] = None
        sampling.initialize(X_train, y_train)        
        for c_idx in idx_test:
            print(c_idx, end="\r")
            new_X = X_test.loc[[c_idx], columns_x]
            new_y = y_test.loc[[c_idx-delay],[columns_y]]
            sampling.add(new_X, new_y)
            X,y = sampling.get_sample()
            
            model.fit(X.values, y.values.flatten())
            
            if name=="TD":
                bef_idx = c_idx - sampling.delta #前時刻データのインデックス                             
                results.loc[c_idx, "y_pred_{}_{}".format(name, key)] = \
                    model.predict(new_X-sampling.X.iloc[bef_idx]).flatten()+sampling.y.iloc[bef_idx].values
            else:
                results.loc[c_idx, "y_pred_{}_{}".format(name, key)] = model.predict(new_X).flatten() #new_Xに該当するyを予測
        print("done")

results["measured"] = df.loc[idx_test,[columns_y]] #実測値をdfに追加
import os 
if not os.path.exists("./result/"):
    os.mkdir("result")

with open("result/new.txt","w") as f:
    for i in range(results.shape[1]-1):
        f.write("{} : {}\n".format(results.columns[i], r2_score(results.iloc[:,i], results.iloc[:,-1])))



results = results.astype(float)
results = results.round(5)
results.to_csv("result/dataframe.csv")

plt.figure()
for i in range(results.shape[1]):
    plt.plot(results.index, results[results.columns[i]],label=results.columns[i])
plt.xlabel("index")
plt.ylabel("y2")
plt.legend(loc="best")
plt.savefig("result/plot.png")   

with open("result/new.txt","w") as f:
    for i in range(results.shape[1]-1):
        f.write("{} : {}\n".format(results.columns[i], r2_score(results.iloc[:,i], results.iloc[:,-1])))
    t2 = time.time()
    f.write("実行時間 : {}".format(t2-t1))