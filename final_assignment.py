from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

DATASET_PATH = "Data/dublinbikes_20200101_20200401.csv"
SELECTED_STATIONS = [10, 96]

def plot_station_data(df_station_time_in_days, y_available_bikes, station_id):
    # plot number of available bikes at station id vs number of days
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(figsize=(8, 8), dpi=80)
    plt.plot(df_station_time_in_days, y_available_bikes,'-o')
    plt.xlabel('Time (Days)')
    plt.ylabel(f'Available Bikes at Station {station_id}')
    plt.title(f'Available bikes vs Days for bike station {station_id}')
    plt.show()


def first_exec(df_station, station_id):
    # convert date/time to unix timestamp in sec
    df_station_time_in_sec = (pd.array((pd.DatetimeIndex(df_station.iloc[:, 1])).astype(np.int64)) / 1000000000)
    time_sampling_interval = df_station_time_in_sec[1] - df_station_time_in_sec[0]
    print("data sampling interval is %d secs" % time_sampling_interval)

    df_station_time_in_days = ((df_station_time_in_sec - df_station_time_in_sec[0]) / 60 / 60 / 24)  # convert timestamp to days
    y_available_bikes = np.extract(df_station_time_in_sec, df_station.iloc[:, 6]).astype(np.int64)
    plot_station_data(df_station_time_in_days, y_available_bikes, station_id)


def main():
    df_dublin_bikes = pd.read_csv(DATASET_PATH)
    df_dublin_bikes["TIME"] = pd.to_datetime(df_dublin_bikes["TIME"])
    df_dublin_bikes.rename(columns={"STATION ID": "STATION_ID"}, inplace=True)
    for station in SELECTED_STATIONS:
        station_id = df_dublin_bikes.STATION_ID == station
        df_station = df_dublin_bikes[station_id]
        first_exec(df_station, station)
        break


if __name__ == '__main__':
    main()
