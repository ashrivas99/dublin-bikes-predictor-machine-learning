import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

DATASET_PATH = "Data/dublinbikes_20200101_20200401.csv"
SELECTED_STATIONS = [19, 10, 96]


def plot_station_data(time_full_days, y_available_bikes, station_id):
    # plot number of available bikes at station id vs number of days
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(figsize=(8, 8), dpi=80)
    # plt.plot(time_full_days, y_available_bikes, '-o')
    plt.scatter(time_full_days, y_available_bikes, color="red", marker=".")
    plt.xlabel("Time (Days)")
    plt.ylabel(f"Available Bikes at Station {station_id}")
    plt.title(f"Available bikes vs Days for bike station {station_id}")
    plt.show()


def plot_preds(time_full_days, y_available_bikes, time_preds_days, y_pred, station_id):
    plt.scatter(time_full_days, y_available_bikes, color="black")
    plt.scatter(time_preds_days, y_pred, color="blue")
    plt.xlabel("Time (days)")
    plt.ylabel(f"Available Bikes at Station {station_id}")
    plt.title(f"Available bikes vs predicted available bikes ")
    plt.legend(["training data", "predictions"], loc="upper right")
    plt.show()


def predidct_available_bikes_seasonality(
    q_step_ahead,
    seasonality,
    lag,
    y_available_bikes,
    time_full_days,
    time_sampling_interval_dt,
    plot,
    station_id,
    trend,
):
    # q_step_ahead-step ahead prediction
    stride = 1
    XX_input_features = y_available_bikes[
        0 : y_available_bikes.size - q_step_ahead - lag * seasonality : stride
    ]

    for i in range(1, lag):
        X = y_available_bikes[
            i * seasonality : y_available_bikes.size
            - q_step_ahead
            - (lag - i) * seasonality : stride
        ]
        XX_input_features = np.column_stack((XX_input_features, X))

    yy_outputs_available_bikes = y_available_bikes[
        lag * seasonality + q_step_ahead :: stride
    ]
    time_for_each_prediction_in_days = time_full_days[
        lag * seasonality + q_step_ahead :: stride
    ]

    train, test = train_test_split(
        np.arange(0, yy_outputs_available_bikes.size), test_size=0.2
    )

    model = Ridge(fit_intercept=False).fit(
        XX_input_features[train], yy_outputs_available_bikes[train]
    )
    print(model.intercept_, model.coef_)
    y_pred = model.predict(XX_input_features)

    if plot:
        plt.rc("font", size=18)
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.scatter(time_full_days, y_available_bikes, color="black")
        plt.scatter(time_for_each_prediction_in_days, y_pred, color="blue")
        plt.xlabel("Time (days)")
        plt.ylabel(f"Available Bikes at Station {station_id}")
        plt.title(f"Available bikes vs predicted available bikes using {trend}")
        plt.legend(["training data", "predictions"], loc="upper right")
        num_samples_per_day = math.floor(24 * 60 * 60 / time_sampling_interval_dt)
        # plt.xlim(((lag*seasonality+q_step_ahead)/num_samples_per_day,(lag*seasonality+q_step_ahead)/num_samples_per_day+10))
        plt.show()


def test_various_seasonaliy_preds(
    y_available_bikes, time_full_days, time_sampling_interval_dt, station_id
):
    plot = True

    # prediction using short-term trend
    predidct_available_bikes_seasonality(
        q_step_ahead=10,
        seasonality=1,
        lag=3,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
        plot=plot,
        station_id=station_id,
        trend="Short Term Trend",
    )

    # prediction using daily seasonality
    d = math.floor(
        24 * 60 * 60 / time_sampling_interval_dt
    )  # number of samples per day
    predidct_available_bikes_seasonality(
        q_step_ahead=d,
        seasonality=d,
        lag=3,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
        plot=plot,
        station_id=station_id,
        trend="Daily Seasonality",
    )

    # prediction using weekly seasonality
    w = math.floor(
        7 * 24 * 60 * 60 / time_sampling_interval_dt
    )  # number of samples per day
    predidct_available_bikes_seasonality(
        q_step_ahead=w,
        seasonality=w,
        lag=3,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
        plot=plot,
        station_id=station_id,
        trend="Weekly Seasonality",
    )


def feature_engineering(
    q_step_size,
    lag,
    stride,
    y_available_bikes,
    time_full_days,
    time_sampling_interval_dt,
):
    # number of samples per day
    num_samples_per_day = math.floor(24 * 60 * 60 / time_sampling_interval_dt)
    # number of samples per week
    num_samples_per_week = math.floor(7 * 24 * 60 * 60 / time_sampling_interval_dt)

    len = (
        y_available_bikes.size
        - num_samples_per_week
        - lag * num_samples_per_week
        - q_step_size
    )
    XX_input_features = y_available_bikes[q_step_size : q_step_size + len : stride]

    for i in range(1, lag):
        X = y_available_bikes[
            i * num_samples_per_week
            + q_step_size : i * num_samples_per_week
            + q_step_size
            + len : stride
        ]
        XX_input_features = np.column_stack((XX_input_features, X))

    for i in range(0, lag):
        X = y_available_bikes[
            i * num_samples_per_day
            + q_step_size : i * num_samples_per_day
            + q_step_size
            + len : stride
        ]
        XX_input_features = np.column_stack((XX_input_features, X))

    for i in range(0, lag):
        X = y_available_bikes[i : i + len : stride]
        XX_input_features = np.column_stack((XX_input_features, X))

    yy_outputs_available_bikes = y_available_bikes[
        lag * num_samples_per_week
        + num_samples_per_week
        + q_step_size : lag * num_samples_per_week
        + num_samples_per_week
        + q_step_size
        + len : stride
    ]
    time_for_each_prediction_in_days = time_full_days[
        lag * num_samples_per_week
        + num_samples_per_week
        + q_step_size : lag * num_samples_per_week
        + num_samples_per_week
        + q_step_size
        + len : stride
    ]

    return (
        XX_input_features,
        yy_outputs_available_bikes,
        time_for_each_prediction_in_days,
    )


def exam_2021(df_station, station_id):
    # converting timestamp to unix timestamp in seconds
    time_full_seconds = (
        pd.array(pd.DatetimeIndex(df_station.iloc[:, 1]).astype(np.int64)) / 1000000000
    )
    time_full_seconds = time_full_seconds.to_numpy()
    time_sampling_interval_dt = time_full_seconds[1] - time_full_seconds[0]
    print(
        f"data sampling interval is {time_sampling_interval_dt} secs or {time_sampling_interval_dt/60} minutes"
    )

    time_full_days = (
        (time_full_seconds - time_full_seconds[0]) / 60 / 60 / 24
    )  # convert timestamp to days
    y_available_bikes = np.extract([time_full_seconds], df_station.iloc[:, 6]).astype(
        np.int64
    )

    # plot extracted data
    plot_station_data(time_full_days, y_available_bikes, station_id)
    test_various_seasonaliy_preds(
        y_available_bikes, time_full_days, time_sampling_interval_dt, station_id
    )

    XX, yy, time_preds_days = feature_engineering(
        q_step_size=2,
        lag=3,
        stride=1,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
    )

    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)

    model = Ridge(fit_intercept=False).fit(XX[train], yy[train])
    print(model.intercept_, model.coef_)
    y_pred = model.predict(XX)

    plot_preds(time_full_days, y_available_bikes, time_preds_days, y_pred, station_id)


def main():
    df_dublin_bikes = pd.read_csv(DATASET_PATH)
    df_dublin_bikes["TIME"] = pd.to_datetime(df_dublin_bikes["TIME"])
    df_dublin_bikes.rename(columns={"STATION ID": "STATION_ID"}, inplace=True)
    for station in SELECTED_STATIONS:
        # station_id = df_dublin_bikes.STATION_ID == station
        # df_station = df_dublin_bikes[station_id]
        df_station = df_dublin_bikes.loc[df_dublin_bikes["STATION_ID"] == station]
        exam_2021(df_station, station)
        break


if __name__ == "__main__":
    main()
