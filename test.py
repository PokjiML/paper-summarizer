import pandas as pd

test_df = pd.read_csv('data/train.csv')
print(test_df['source'][100])