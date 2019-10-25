
class EarlyStopping():
    def __init__(self,patience=0,verbose=0):
        self._step = 0
        self._loss = float("inf")
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss): #　self._lossを上回る誤差の回数が連続でself.patience回よりも多ければEarlyStopping
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                print("early stopping")
            return True
        else:
            self._step = 0
            self._loss = loss

        return False