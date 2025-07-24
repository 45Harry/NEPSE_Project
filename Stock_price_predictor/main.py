from stock import  stock_data
import pandas as pd
import numpy as np


def data_frame(raw_data):
    return pd.DataFrame(raw_data)

if __name__ == '__main__':
    raw_data = stock_data('NEPSE')
    df = data_frame(raw_data)


    df.to_csv('output.csv', index=False)  # index=False avoids writing row numbers
    print(df)
