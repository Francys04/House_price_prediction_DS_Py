import pandas as pd

data = 'C:\\Users\\Phantom\\Desktop\\Python project\\DS\\House-price prediction\\data\\boston.csv'

load_boston = pd.read_csv(data)
print(load_boston)