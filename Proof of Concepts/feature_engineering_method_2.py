import math
from operator import index
from matplotlib.style import available

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

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


def regression_evaluation_metircs(ytest, ypred):
    mean_abs_err = metrics.mean_absolute_error(ytest, ypred)
    mean_sq_err = metrics.mean_squared_error(ytest, ypred)
    root_mean_sq_err = np.sqrt(mean_sq_err)
    median_abs_err = metrics.median_absolute_error(ytest, ypred)
    r2Score = metrics.r2_score(ytest, ypred)

    scores_dict = {
        "mean_sq_err": mean_sq_err,
        "root_mean_sq_err": root_mean_sq_err,
        "mean_abs_err": mean_abs_err,
        "r2Score": r2Score,
        "median_abs_err": median_abs_err,
    }

    print(f"Mean Squared Error: {mean_sq_err} ")
    print(f"Root Mean Squared Error: {root_mean_sq_err}")
    print(f"Mean_Absolute_Error: {mean_abs_err}")
    print(f"R2 Score: {r2Score}")
    print(f"Median Absolute Error: {median_abs_err}")

    return scores_dict


def feature_engineering1(
    df_station,
    lag,
    q_step_sizeez,
    time_sampling_interval_dt,
    short_term_features_flag,
    daily_features_flag,
    weekly_features_flag,
):

    available_bikes_df = df_station[["AVAILABLE BIKES"]]
    available_bikes_df = available_bikes_df.dropna()

    time_sampling_interval_dt_mins = time_sampling_interval_dt / 60
    # Setting outputs y for step ahead predictions
    for step_size in q_step_sizeez:
        q = step_size * time_sampling_interval_dt_mins
        available_bikes_df.loc[
            :, f"bikes_avail_{q}_mins_ahead"
        ] = available_bikes_df.loc[:, "AVAILABLE BIKES"].shift(-step_size)

    if short_term_features_flag:
        for data_point in range(0, lag):
            available_bikes_df.loc[
                :, f"Bikes_{data_point+1}_point_before"
            ] = available_bikes_df.loc[:, "AVAILABLE BIKES"].shift(data_point + 1)

    # number of samples per day
    num_samples_per_day = math.floor(24 * 60 * 60 / time_sampling_interval_dt)

    if daily_features_flag:
        for day in range(0, lag):
            available_bikes_df.loc[
                :, f"Bikes_{day+1}_day_ago"
            ] = available_bikes_df.loc[:, "AVAILABLE BIKES"].shift(
                num_samples_per_day * (day + 1)
            )

    # number of samples per week
    num_samples_per_week = math.floor(7 * 24 * 60 * 60 / time_sampling_interval_dt)

    if weekly_features_flag:
        for week in range(0, lag):
            available_bikes_df.loc[
                :, f"Bikes_{week+1}_week_ago"
            ] = available_bikes_df.loc[:, "AVAILABLE BIKES"].shift(
                num_samples_per_week * (week + 1)
            )

    time_series_features_10_min_ahead_preds = available_bikes_df.copy()
    time_series_features_10_min_ahead_preds = time_series_features_10_min_ahead_preds.drop(
        ['AVAILABLE BIKES','bikes_avail_30.0_mins_ahead', 'bikes_avail_60.0_mins_ahead'], axis=1
    )

    time_series_features_30_min_ahead_preds = available_bikes_df.copy()
    time_series_features_30_min_ahead_preds = time_series_features_30_min_ahead_preds.drop(
        ['AVAILABLE BIKES','bikes_avail_10.0_mins_ahead', 'bikes_avail_60.0_mins_ahead'], axis=1
    )

    time_series_features_1hr_ahead_preds = available_bikes_df.copy()
    time_series_features_1hr_ahead_preds = time_series_features_1hr_ahead_preds.drop(
        ['AVAILABLE BIKES','bikes_avail_10.0_mins_ahead', 'bikes_avail_30.0_mins_ahead'], axis=1
    )

    time_series_features_10_min_ahead_preds = time_series_features_10_min_ahead_preds.dropna()
    time_series_features_30_min_ahead_preds = time_series_features_30_min_ahead_preds.dropna()
    time_series_features_1hr_ahead_preds = time_series_features_1hr_ahead_preds.dropna()
    available_bikes_df = available_bikes_df.dropna()

    return time_series_features_10_min_ahead_preds, time_series_features_30_min_ahead_preds, time_series_features_1hr_ahead_preds

def lagCrossValidation( df_station, q_step_size, time_sampling_interval_dt, station_id):
    mean_error = []; std_error = []
    lag_range = list(range(2, 50))
    test_model_ridge = Ridge(fit_intercept=False); test_model_kNNR = KNeighborsRegressor()

    models = [test_model_ridge, test_model_kNNR]
    for model in models:
        for lag_value in lag_range:
            df_10_min_ahead_preds, df_30_min_ahead_preds, df_1hr_ahead_preds = feature_engineering(
            df_station=df_station,
            lag=lag_value,
            q_step_size=q_step_size,
            time_sampling_interval_dt=time_sampling_interval_dt,
            short_term_features_flag=True,
            daily_features_flag=True,
            weekly_features_flag=True,
        )



            temp = []
            kf = KFold(n_splits=10)
            for train, test in kf.split(XX_test):
                model.fit(XX_test[train], yy_test[train])
                ypred = model.predict(XX_test[test])
                temp.append(mean_squared_error(yy_test[test], ypred))
            # plot_preds(time_full_days, y_available_bikes, time_preds_days,  model.predict(XX_test), station_id)
            mean_error.append(np.array(temp).mean())
            std_error.append(np.array(temp).std())
        plt.rc("font", size=18)
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.errorbar(lag_range, mean_error, yerr=std_error, linewidth=3)
        plt.xlabel("lag")
        plt.ylabel("negative mean squared error")
        plt.title(f"Lag Cross Validation Results,{q_value*(time_sampling_interval_dt/60)} minutes ahead Preidctions")
        plt.show()

def exam_2021(df_station, station_id):
    start_date = pd.to_datetime("29-01-2020", format="%d-%m-%Y")
    df_station = df_station[df_station.TIME > start_date]

    # Finding the time interval between each measurement
    time_full_seconds = (
        pd.array(pd.DatetimeIndex(df_station.iloc[:, 1]).astype(np.int64)) / 1000000000
    )
    time_full_seconds = time_full_seconds.to_numpy()
    time_sampling_interval_dt = time_full_seconds[1] - time_full_seconds[0]
    print(
        f"data sampling interval is {time_sampling_interval_dt} secs or {time_sampling_interval_dt/60} minutes"
    )

    # Extracting all data for available bikes into df_total_station_data
    df_station = df_station.set_index("TIME")
    df_total_station_data = df_station[["AVAILABLE BIKES"]]
    df_total_station_data = df_total_station_data.dropna()

    # Plotting avaliable bikes vs time of our dataset
    plot_station_data(
        df_total_station_data.index, df_total_station_data["AVAILABLE BIKES"], 19
    )

    # Setting the step sizes
    step_size_10_min_ahead_preds = 2
    step_size_30_min_ahead_preds = 6
    step_size_1_hr_ahead_preds = 12
    q_step_sizeez = [
        2,
        6,
        12
    ]

    # Calculating features for step ahead predictions and extracting these features into seperate dataframes
    df_10_min_ahead_preds, df_30_min_ahead_preds, df_1hr_ahead_preds = feature_engineering1(
        df_station=df_station,
        lag=3,
        q_step_sizeez=q_step_sizeez,
        time_sampling_interval_dt=time_sampling_interval_dt,
        short_term_features_flag=True,
        daily_features_flag=True,
        weekly_features_flag=True,
    )


    # Setting up Hold Out train and test data for predicting available bikes 10 minutes ahead
    train_indices_df_10_min = 0.70 * df_10_min_ahead_preds.shape[0]
    XX_10_min = df_10_min_ahead_preds.drop(["bikes_avail_10.0_mins_ahead"], axis=1)
    
    X_train_10_min = df_10_min_ahead_preds[: int(train_indices_df_10_min)].drop(["bikes_avail_10.0_mins_ahead"], axis=1)
    y_train_10_min = df_10_min_ahead_preds[: int(train_indices_df_10_min)]
    y_train_10_min = y_train_10_min[["bikes_avail_10.0_mins_ahead"]]

    X_test_10_min = df_10_min_ahead_preds[int(train_indices_df_10_min) :].drop(["bikes_avail_10.0_mins_ahead"], axis=1)
    y_test_10_min = df_10_min_ahead_preds[int(train_indices_df_10_min) :]
    y_test_10_min = y_test_10_min[["bikes_avail_10.0_mins_ahead"]]


    # Setting up Hold Out train and test data for predicting available bikes 30 minutes ahead
    train_indices_df_30_min = 0.70 * df_30_min_ahead_preds.shape[0]
    XX_30_min = df_30_min_ahead_preds.drop(["bikes_avail_30.0_mins_ahead"], axis=1)
    
    X_train_30_min = df_30_min_ahead_preds[: int(train_indices_df_30_min)].drop(["bikes_avail_30.0_mins_ahead"], axis=1)
    y_train_30_min = df_30_min_ahead_preds[: int(train_indices_df_30_min)]
    y_train_30_min = y_train_30_min[["bikes_avail_30.0_mins_ahead"]]

    X_test_30_min = df_30_min_ahead_preds[int(train_indices_df_30_min) :].drop(["bikes_avail_30.0_mins_ahead"], axis=1)
    y_test_30_min = df_30_min_ahead_preds[int(train_indices_df_30_min) :]
    y_test_30_min = y_test_30_min[["bikes_avail_30.0_mins_ahead"]]


    # Setting up Hold Out train and test data for predicting available bikes 1 hour ahead
    train_indices_df_1hr = 0.70 * df_1hr_ahead_preds.shape[0]
    XX_1hr = df_1hr_ahead_preds.drop(["bikes_avail_60.0_mins_ahead"], axis=1)
    
    X_train_1hr = df_1hr_ahead_preds[: int(train_indices_df_1hr)].drop(["bikes_avail_60.0_mins_ahead"], axis=1)
    y_train_1hr = df_1hr_ahead_preds[: int(train_indices_df_1hr)]
    y_train_1hr = y_train_1hr[["bikes_avail_60.0_mins_ahead"]]

    X_test_1hr = df_1hr_ahead_preds[int(train_indices_df_1hr) :].drop(["bikes_avail_60.0_mins_ahead"], axis=1)
    y_test_1hr = df_1hr_ahead_preds[int(train_indices_df_1hr) :]
    y_test_1hr = y_test_1hr[["bikes_avail_60.0_mins_ahead"]]




    # _____

    model = Ridge()
    param_search = {"alpha": [1, 10]}
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(
        estimator=model, cv=tscv, param_grid=param_search, scoring=rmse_score
    )
    gsearch.fit(X_train, y_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    y_true = y_test.values
    y_pred = best_model.predict(X_test)
    regression_evaluation_metircs(y_true, y_pred)
    plot_preds(
        available_bikes_df1.index,
        available_bikes_df1["AVAILABLE BIKES"],
        XX.index,
        best_model.predict(XX),
        19,
    )

    reg = RidgeCV()
    reg.fit(X_train, y_train)
    print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    print("Best score using built-in RidgeCV: %f" % reg.score(X_train, y_train))
    weights = reg.coef_.reshape(9)
    coef = pd.Series(weights, index=X_train.columns.to_numpy())
    print(
        "Ridge picked "
        + str(sum(coef != 0))
        + " variables and eliminated the other "
        + str(sum(coef == 0))
        + " variables"
    )
    imp_coef = coef.sort_values()
    import matplotlib

    matplotlib.rcParams["figure.figsize"] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Feature importance using Ridge Model")
    plt.show()


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

