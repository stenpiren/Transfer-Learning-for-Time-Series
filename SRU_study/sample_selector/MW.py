

class MW:
    def __init__(self, n_window):
        self.n_window = n_window
        
    def initialize(self, X, y):
        self.X = X
        self.y = y
        
    def add(self, new_X, new_y):
        self.X = self.X.append(new_X)
        self.y = self.y.append(new_y)
        
    def get_sample(self):
        index = sorted( set(self.X.index) & set(self.y.index) )[-self.n_window:]
        return self.X.loc[index,:], self.y.loc[index,:]