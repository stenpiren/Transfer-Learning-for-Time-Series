import numpy as np 
import pandas as pd

class TD:    
    def __init__(self, n_window, delta=9):
        self.n_window = n_window
        #時間差分の間隔self.deltaは遅れ時間delayよりも大きい値をとらなければならない
        self.delta = delta 

    def initialize(self, X, y):
        self.X = X
        self.y = y
        
    def add(self, new_X, new_y):
        self.X = self.X.append(new_X)
        self.y = self.y.append(new_y)
        
    def get_sample(self):
        index = sorted( set(self.X.index) & set(self.y.index) )[-self.n_window:]
        diff_X = pd.DataFrame()
        diff_y = pd.DataFrame()
        for i in index: 
            diff_X = diff_X.append(self.X.iloc[i]-self.X.iloc[i-self.delta],ignore_index=True)
            diff_y = diff_y.append(self.y.iloc[i]-self.y.iloc[i-self.delta],ignore_index=True)                
        return diff_X, diff_y 