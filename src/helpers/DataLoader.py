import pandas as pd


class Dataloader:
    def __init__(self,path):
        self.path = path

    def preprocessdata(self, data):
        pass
        
    def load_data(self.path):
        data = pd.read_csv(self.path)
        return data
        
   