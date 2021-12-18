import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.dummy import DummyRegressor

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
    model,
    q_step_ahead,
    seasonality,
    lag,
    y_available_bikes,
    time_full_days,
    time_sampling_interval_dt,
    plot,
    station_id,
    trend,
    print_flag,
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

    model_info = model.fit(XX_input_features[train], yy_outputs_available_bikes[train])
    if print_flag:
        print(model_info.intercept_, model_info.coef_)

    y_pred = model_info.predict(XX_input_features)

    mse = mean_squared_error(
        yy_outputs_available_bikes[test],
        model_info.predict(XX_input_features[test]),
        multioutput="uniform_average",
        squared=True,
    )
    print(f"Ridge MSE: {mse}")

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
    test_model, y_available_bikes, time_full_days, time_sampling_interval_dt, station_id
):
    plot = True
    print_flag = True
    # prediction using short-term trend
    predidct_available_bikes_seasonality(
        model=test_model,
        q_step_ahead=10,
        seasonality=1,
        lag=3,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
        plot=plot,
        station_id=station_id,
        trend="Short Term Trend",
        print_flag=print_flag,
    )

    # prediction using daily seasonality
    d = math.floor(
        24 * 60 * 60 / time_sampling_interval_dt
    )  # number of samples per day
    predidct_available_bikes_seasonality(
        model=test_model,
        q_step_ahead=2,
        seasonality=d,
        lag=3,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
        plot=plot,
        station_id=station_id,
        trend="Daily Seasonality",
        print_flag=print_flag,
    )

    # prediction using weekly seasonality
    w = math.floor(
        7 * 24 * 60 * 60 / time_sampling_interval_dt
    )  # number of samples per day
    predidct_available_bikes_seasonality(
        model=test_model,
        q_step_ahead=2,
        seasonality=w,
        lag=3,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
        plot=plot,
        station_id=station_id,
        trend="Weekly Seasonality",
        print_flag=print_flag,
    )


def feature_engineering(
    q_step_size,
    lag,
    stride,
    y_available_bikes,
    time_full_days,
    time_sampling_interval_dt,
    weekly_features_flag,
    daily_features_flag,
    short_term_features_flag,
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

    if weekly_features_flag:
        for i in range(1, lag):
            X = y_available_bikes[
                i * num_samples_per_week
                + q_step_size : i * num_samples_per_week
                + q_step_size
                + len : stride
            ]
            XX_input_features = np.column_stack((XX_input_features, X))

    if daily_features_flag:
        for i in range(0, lag):
            X = y_available_bikes[
                i * num_samples_per_day
                + q_step_size : i * num_samples_per_day
                + q_step_size
                + len : stride
            ]
            XX_input_features = np.column_stack((XX_input_features, X))

    if short_term_features_flag:
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


def PolynomialOrderCrossValidation(XX, yy):
    kf = KFold(n_splits=10)
    mean_error = []
    std_error = []
    q_range = [1, 2, 3, 4, 5]
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(XX)
        model_ridge = Ridge()
        temp = []
        for train, test in kf.split(XX):
            model_ridge.fit(Xpoly[train], yy[train])
            ypred = model_ridge.predict(Xpoly[test])
            temp.append(mean_squared_error(yy[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("q")
    plt.ylabel("Mean square error")
    plt.title("Ridge Regression Cross Validation Results: Polynomial Feature q")
    plt.show()


def RidgeAlphaValueCrossValidation(X, y):
    mean_error = []
    std_error = []
    C_range = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000]
    for C in C_range:
        ridge_model = Ridge(alpha=1 / (2 * C))
        temp = []
        kf = KFold(n_splits=10)
        for train, test in kf.split(X):
            ridge_model.fit(X[train], y[train])
            ypred = ridge_model.predict(X[test])
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(C_range, mean_error, yerr=std_error)
    plt.xlabel("Ci")
    plt.ylabel("Mean square error")
    plt.title("Ridge Regression Cross Validation for a range of C")
    # plt.xlim((0, 200))
    plt.show()


def KNeighborsRegressor_k_value_CV(XX, yy):
    mean_error = []
    std_error = []
    num_neighbours = list(range(1, 50))
    for k in num_neighbours:
        model = KNeighborsRegressor(n_neighbors=k, weights="uniform")
        scores = cross_val_score(model, XX, yy, cv=10, scoring="neg_mean_squared_error")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(num_neighbours, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("k: number of neighbours")
    plt.ylabel("negative mean squared error")
    plt.title("KNeighborsRegressor Cross Validation Results")
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


def ridgeRegression(XX, yy, train, test, C_value, time_preds_days, station_id):
    ridgeRegression_model = Ridge(alpha=(1 / (2 * C_value)), fit_intercept=False).fit(
        XX[train], yy[train]
    )
    print("Ridge Regression Model Parameters")
    print(ridgeRegression_model.intercept_, ridgeRegression_model.coef_)
    ypred = ridgeRegression_model.predict(XX[test])

    ridge_metrics_scores = regression_evaluation_metircs(yy[test], ypred)

    # print('Plotting y_test vs y_predictions')
    # plot_preds(time_preds_days[test], yy[test], time_preds_days[test], ypred, station_id)

    return ridgeRegression_model, ridge_metrics_scores


def kNearestNeighborsRegression(
    XX, yy, train, test, num_neighbors_k, time_preds_days, station_id
):
    kNR_model = KNeighborsRegressor(n_neighbors=num_neighbors_k, weights="uniform")
    kNR_model.fit(XX[train], yy[train])
    ypred = kNR_model.predict(XX[test])
    ridge_metrics_scores = regression_evaluation_metircs(yy[test], ypred)
    # print('Plotting y_test vs y_predictions')
    # plot_preds(time_preds_days[test], yy[test], time_preds_days[test], ypred, station_id)

    return kNR_model, ridge_metrics_scores


def returnSameDataPoint(yy):
    return yy


# Baseline model to predict same point as the last
def baselineModel(
    XX, yy, test, time_preds_days, station_id, time_full_days, y_available_bikes
):
    y_pred_test = []
    ytest = yy[test]
    for index, y_available_bike in enumerate(ytest):
        if index > 0:
            y_pred = returnSameDataPoint(ytest[index - 1])
            y_pred_test.append(y_pred)
        else:
            y_pred_test.append(0)
    y_pred_test = np.array(y_pred_test)
    baseline_metrics_scores = regression_evaluation_metircs(yy[test], y_pred_test)

    print("Plotting Baseline Regression Predictions")
    plot_preds(
        time_full_days,
        y_available_bikes,
        time_preds_days[test],
        y_pred_test,
        station_id,
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

    test_model_ridge = Ridge(fit_intercept=False)
    test_various_seasonaliy_preds(
        test_model_ridge,
        y_available_bikes,
        time_full_days,
        time_sampling_interval_dt,
        station_id,
    )

    XX, yy, time_preds_days = feature_engineering(
        q_step_size=2,
        lag=3,
        stride=1,
        y_available_bikes=y_available_bikes,
        time_full_days=time_full_days,
        time_sampling_interval_dt=time_sampling_interval_dt,
        weekly_features_flag=True,
        daily_features_flag=True,
        short_term_features_flag=False,
    )

    # -----------------------------------------Cross Validation---------------------------------------
    scaler = MinMaxScaler()
    XX_scaled = scaler.fit_transform(XX)

    # Polynomial Order Cross Validation for Ridge Regression
    PolynomialOrderCrossValidation(XX_scaled, yy)
    polynomial_order_ridge = int(
        input("Please enter the polynomial order for Ridge Regression 'q' value:    ")
    )

    XX_polynomial = XX
    if polynomial_order_ridge > 1:
        XX_polynomial = PolynomialFeatures(polynomial_order_ridge).fit_transform(
            XX_scaled
        )

    RidgeAlphaValueCrossValidation(XX_polynomial, yy)
    C_value_ridge = int(
        input("Please choose the desired 'C' value for the Ridge Regression model:    ")
    )

    KNeighborsRegressor_k_value_CV(XX, yy)
    k_value = int(
        input(
            "Please enter the number of neighnours 'k' value for the KNeighborsRegressor model:    "
        )
    )

    # -----------------------------------------Predicting available bikes using Ridge and KNeigborsRegressor---------------------------------------

    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
    # -----------------------------------------Ridge Regression---------------------------------------
    model_ridgeReg, scores_ridgeReg = ridgeRegression(
        XX_polynomial, yy, train, test, C_value_ridge, time_preds_days, station_id
    )
    ypred_full_ridge = model_ridgeReg.predict(XX)

    print("Plotting Ridge Regression Predictions")
    plot_preds(
        time_full_days, y_available_bikes, time_preds_days, ypred_full_ridge, station_id
    )
    print(scores_ridgeReg)

    # -----------------------------------------KNeigborsRegressor---------------------------------------
    model_kNR, scores_kNR = kNearestNeighborsRegression(
        XX_polynomial, yy, train, test, k_value, time_preds_days, station_id
    )
    ypred_full_kNR = model_kNR.predict(XX)

    print("Plotting Ridge Regression Predictions")
    plot_preds(
        time_full_days, y_available_bikes, time_preds_days, ypred_full_kNR, station_id
    )
    print(scores_kNR)

    # ----------------------------------------------Baseline-------------------------------------------------
    baselineModel(
        XX_polynomial,
        yy,
        test,
        time_preds_days,
        station_id,
        time_full_days,
        y_available_bikes,
    )


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

# TODO:
# do model features mean square error -> ss/ ss+d/ ss+w/ d/ d+w/ w
# selecting lag cross validation i.e how many points before

# Train models using cross validation -> did not go well for Ridge
# Dummy Model -> predicting same point as the last is better idea


# NOTE:
# let the data help tell you which features are important
# e.g. by fitting a linear model and looking at the weights given to each feature and/or by progressively adding/removing features
# to see the impact on predictions.
# When using linear models its often worth looking at use of augmented features e.g. polynomials (but not just that).
