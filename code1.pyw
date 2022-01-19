import pandas as pd

df=pd.read_csv("ML/FINALPROJECT/USB-IDS-1-TRAINING.csv", header=None, low_memory=False)

df = df.sample(n=10000)

df.to_csv('./RandomSample.csv', index=False)
