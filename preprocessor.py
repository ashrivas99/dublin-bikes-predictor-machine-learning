import pandas as pd
import glob
import os

DATASET_PATH = "Data/dublinbikes_20200101_20200401.csv"


def main():
    df_bikes_2020_q1_usage = pd.read_csv(DATASET_PATH)
    print(df_bikes_2020_q1_usage.shape)


if __name__ == '__main__':
    main()
