import numpy as np 
import pandas as pd

class JIT:
    def __init__(self, n_window):
        self.n_window = n_window
        
    def initialize(self, X, y):
        self.X = X
        self.y = y
        
    def add(self, new_X, new_y):
        self.X = self.X.append(new_X)
        self.y = self.y.append(new_y)
        
    def get_sample(self):
        base = self.y.iloc[-1].values # 基準となるデータ
        distance_list = np.empty([0], dtype=float) #空の距離リストを作成
        for i in range(self.y.shape[0]-1):
            now = self.y.iloc[i].values
            distance = np.linalg.norm(base - now)
            distance_list = np.append(distance_list, distance)
        index_list = np.argsort(distance_list)[:self.n_window]  
        return self.X.loc[index_list,:], self.y.loc[index_list,:] # indexの値をすべて含めたいのでilocでなくlocを使う
