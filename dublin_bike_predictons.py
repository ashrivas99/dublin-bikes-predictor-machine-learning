import math
from operator import index

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.style import available
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, KFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
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


def feature_engineering(
    df_station,
    lag,
    q_step_size,
    time_sampling_interval_dt,
    short_term_features_flag,
    daily_features_flag,
    weekly_features_flag,
):
    available_bikes_df = df_station[["AVAILABLE BIKES"]]
    available_bikes_df = available_bikes_df.dropna()

    time_sampling_interval_dt_mins = time_sampling_interval_dt / 60

    # Setting outputs for step ahead predictions
    q = q_step_size * time_sampling_interval_dt_mins

    available_bikes_df.loc[:, f"bikes_avail_{q}_mins_ahead"] = available_bikes_df.loc[
        :, "AVAILABLE BIKES"
    ].shift(-q_step_size)

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

    df_features = available_bikes_df.copy()
    df_features = df_features.drop(["AVAILABLE BIKES"], axis=1)

    df_features = df_features.dropna()
    available_bikes_df = available_bikes_df.dropna()

    # Setting up Hold Out train and test data for predicting available bikes 10 minutes ahead
    XX = df_features.drop([f"bikes_avail_{q}_mins_ahead"], axis=1)
    yy = df_features[[f"bikes_avail_{q}_mins_ahead"]]

    train_indices = 0.70 * df_features.shape[0]
    X_train = df_features[: int(train_indices)].drop(
        [f"bikes_avail_{q}_mins_ahead"], axis=1
    )
    y_train = df_features[: int(train_indices)]
    y_train = y_train[[f"bikes_avail_{q}_mins_ahead"]]

    X_test = df_features[int(train_indices) :].drop(
        [f"bikes_avail_{q}_mins_ahead"], axis=1
    )
    y_test = df_features[int(train_indices) :]
    y_test = y_test[[f"bikes_avail_{q}_mins_ahead"]]

    return XX, yy, X_train, y_train, X_test, y_test, df_features


def lagCrossValidation(
    df_station, time_sampling_interval_dt, df_total_station_data, station_id
):
    mean_error = []
    std_error = []
    lag_range = list(range(1, 6))

    test_model_ridge = Ridge(fit_intercept=False)
    test_model_kNNR = KNeighborsRegressor(n_neighbors=100)
    models = [test_model_ridge, test_model_kNNR]

    step_size = [2, 6, 12]
    for model in models:
        print(f"Model is {model}")
        for q_value in step_size:
            print(f"Step size is {q_value}")
            mean_error = []
            std_error = []
            for lag_value in lag_range:
                XX_CV, yy_CV, X_train, y_train, X_test, y_test, _ = feature_engineering(
                    df_station=df_station,
                    lag=lag_value,
                    q_step_size=q_value,
                    time_sampling_interval_dt=time_sampling_interval_dt,
                    short_term_features_flag=True,
                    daily_features_flag=False,
                    weekly_features_flag=True,
                )
                scores = cross_val_score(
                    model, XX_CV, yy_CV, cv=10, scoring="neg_mean_squared_error"
                )
                mean_error.append(np.array(scores).mean())
                std_error.append(np.array(scores).std())
                # model.fit(X_train, y_train)
                # plot_preds(df_total_station_data.index, df_total_station_data['AVAILABLE BIKES'], XX_CV.index,  model.predict(XX_CV), station_id)
            plt.rc("font", size=18)
            plt.rcParams["figure.constrained_layout.use"] = True
            plt.errorbar(lag_range, mean_error, yerr=std_error, linewidth=3)
            plt.xlabel("lag")
            plt.ylabel("negative mean squared error")
            plt.title(
                f"Lag Cross Validation Results,{q_value*(time_sampling_interval_dt/60)} minutes ahead Preidctions. Model is {model}"
            )
            plt.show()


def featureImportance(df_station, time_sampling_interval_dt, lag_value):
    test_model_ridge = Ridge(fit_intercept=False)
    step_size = [2, 6, 12]
    for q_value in step_size:
        print(f"Step size is {q_value}")
        XX_CV, yy_CV, X_train, y_train, X_test, y_test, _ = feature_engineering(
            df_station=df_station,
            lag=lag_value,
            q_step_size=q_value,
            time_sampling_interval_dt=time_sampling_interval_dt,
            short_term_features_flag=True,
            daily_features_flag=True,
            weekly_features_flag=True,
        )

        test_model_ridge = RidgeCV().fit(X_train, y_train)
        print(
            f"Best alpha value using sklearn Ridge Cross Validation: {test_model_ridge.alpha_}"
        )
        print(
            f"Best negative mse score using sklearn Ridge Cross Validation: {mean_squared_error(yy_CV, test_model_ridge.predict(XX_CV) )}"
        )

        weights = test_model_ridge.coef_.reshape(lag_value * 3)
        weights_with_labels = pd.Series(weights, index=X_train.columns.to_numpy())
        imp_weights_with_labels = weights_with_labels.sort_values()

        # plotting feature importance
        plt.rc("font", size=18)
        plt.rcParams["figure.constrained_layout.use"] = True
        plt.figure(figsize=(8, 8), dpi=80)
        imp_weights_with_labels.plot(kind="barh")
        plt.title(
            f"Feature importance using Ridge Model. Predictions for {q_value*time_sampling_interval_dt/60} minutes ahead"
        )
        plt.show()


def PolynomialOrderCrossValidation(XX_ridge, yy_ridge, XX_kNR, yy_kNR):
    mean_error_ridge = []
    std_error_ridge = []
    mean_error_kNNR = []
    std_error_kNNR = []
    scaler = StandardScaler()
    XX_scaled_ridge = scaler.fit_transform(XX_ridge)
    XX_scaled_kNR = scaler.fit_transform(XX_kNR)
    q_range = [1, 2, 3, 4, 5]
    for q in q_range:
        # Ridge
        Xpoly_ridge = PolynomialFeatures(q).fit_transform(XX_scaled_ridge)
        model_ridge = Ridge()
        scores_ridge = cross_val_score(
            model_ridge, Xpoly_ridge, yy_ridge, cv=10, scoring="neg_mean_squared_error"
        )
        mean_error_ridge.append(np.array(scores_ridge).mean())
        std_error_ridge.append(np.array(scores_ridge).std())

        # KNeighborsRegressor
        Xpoly_kNR = PolynomialFeatures(q).fit_transform(XX_scaled_kNR)
        model_kNNR = KNeighborsRegressor()
        scores_kNNR = cross_val_score(
            model_kNNR, Xpoly_kNR, yy_kNR, cv=10, scoring="neg_mean_squared_error"
        )
        mean_error_kNNR.append(np.array(scores_kNNR).mean())
        std_error_kNNR.append(np.array(scores_kNNR).std())

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(q_range, mean_error_ridge, yerr=std_error_ridge, linewidth=3)
    plt.xlabel("Polynomial Order q")
    plt.ylabel("Negative Mean square error")
    plt.title("Ridge Regression Cross Validation Results: Polynomial Feature q")
    plt.show()

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(q_range, mean_error_kNNR, yerr=std_error_kNNR, linewidth=3)
    plt.xlabel("Polynomial Order q")
    plt.ylabel("Negative Mean square error")
    plt.title("KNeighborsRegressor Cross Validation Results: Polynomial Feature q")
    plt.show()


def RidgeAlphaValueCrossValidation(XX, yy):
    mean_error = []
    std_error = []
    C_range = [
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        5,
        10,
        50,
        100,
        500,
    ]
    for C in C_range:
        ridge_model = Ridge(alpha=1 / (2 * C))
        scores = cross_val_score(
            ridge_model, XX, yy, cv=10, scoring="neg_mean_squared_error"
        )
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(C_range, mean_error, yerr=std_error)
    plt.xlabel("Ci")
    plt.ylabel("Negative Mean square error")
    plt.title("Ridge Regression Cross Validation for a range of C")
    plt.show()


def RidgeAlphaValueCrossValidation_method1(XX, yy):
    mean_error = []
    std_error = []
    XX_local = XX.copy()
    yy_local = yy.copy()
    XX_local = XX_local.reset_index(drop=True)
    yy_local = yy_local.reset_index(drop=True)
    XX_local.index = XX_local.index.astype(int, copy=False)
    yy_local.index = yy_local.index.astype(int, copy=False)
    C_range = [
        0.00001,
        0.00005,
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        5,
        10,
        50,
    ]
    for C in C_range:
        ridge_model = Ridge(alpha=1 / (2 * C))
        temp = []
        kf = KFold(n_splits=10)
        for train, test in kf.split(XX_local):
            ridge_model = Ridge(alpha=1 / (2 * C)).fit(
                XX_local.iloc[train], yy_local.iloc[train]
            )
            ypred = ridge_model.predict(XX_local.iloc[test])
            temp.append(mean_squared_error(yy_local.iloc[test], ypred))
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


def KNeighborsRegressor_k_value_CV(XX, yy, num_neigbor_range):
    mean_error = []
    std_error = []
    num_neighbours = list(range(1, num_neigbor_range))
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


def ridgeRegression(X_train, y_train, X_test, y_test, C_value):
    ridgeRegression_model = Ridge(alpha=1 / (2 * C_value), fit_intercept=False)
    ridgeRegression_model.fit(X_train, y_train)
    print("Ridge Regression Model Parameters")
    print(ridgeRegression_model.intercept_, ridgeRegression_model.coef_)
    ypred = ridgeRegression_model.predict(X_test)
    ridge_metrics_scores = regression_evaluation_metircs(y_test, ypred)
    return ridgeRegression_model, ridge_metrics_scores


def kNearestNeighborsRegression(X_train, y_train, X_test, y_test, num_neighbors_k):
    kNR_model = KNeighborsRegressor(n_neighbors=num_neighbors_k, weights="uniform")
    kNR_model.fit(X_train, y_train)
    ypred = kNR_model.predict(X_test)
    ridge_metrics_scores = regression_evaluation_metircs(y_test, ypred)
    return kNR_model, ridge_metrics_scores


# Baseline model to predict same point as the last TODO
def baselineModel(yy, station_id, pred_point_value):
    yy_baseline_true = yy.copy()
    yy_baseline_true = yy_baseline_true.drop(
        yy_baseline_true.index[range(pred_point_value)]
    )

    yy_baseline_pred = yy.copy()
    yy_baseline_pred.loc[:, f"pred_1_day_before bikes"] = yy_baseline_pred.loc[
        :, yy_baseline_pred.columns[0]
    ].shift(pred_point_value)
    yy_baseline_pred = yy_baseline_pred[yy_baseline_pred.columns[1]]
    yy_baseline_pred = yy_baseline_pred.dropna()

    baseline_metrics_scores = regression_evaluation_metircs(
        yy_baseline_true, yy_baseline_pred
    )

    print("Plotting Baseline Regression Predictions")
    plot_preds(
        yy_baseline_true.index,
        yy_baseline_true[yy_baseline_true.columns[0]],
        yy_baseline_pred.index,
        yy_baseline_pred,
        station_id,
    )


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
        df_total_station_data.index,
        df_total_station_data["AVAILABLE BIKES"],
        station_id,
    )

    # Setting the step sizes
    q_step_size_10_min_ahead_preds = 2
    q_step_size_30_min_ahead_preds = 6
    q_step_size_1_hr_ahead_preds = 12
    q_step_size_list = [
        q_step_size_10_min_ahead_preds,
        q_step_size_30_min_ahead_preds,
        q_step_size_1_hr_ahead_preds,
    ]

    # Performing cross-validation to select the number of prior points, days and weeks to consider.
    lagCrossValidation(
        df_station, time_sampling_interval_dt, df_total_station_data, station_id
    )

    # lag_value_ridge is 4 and lag_value_kNNR is 2
    lag_value_ridge = int(
        input(
            "Please choose the desired 'lag' value for the Ridge Regression model:    "
        )
    )
    lag_value_kNNR = int(
        input(
            "Please choose the desired 'lag' value for the KNeighborsRegressor model:    "
        )
    )

    # Feature Importance for Ridge Regression
    featureImportance(df_station, time_sampling_interval_dt, lag_value_ridge)

    # Calculating features for step ahead predictions and extracting these features into seperate dataframes
    (
        XX_ridge,
        yy_ridge,
        X_train_ridge,
        y_train_ridge,
        X_test_ridge,
        y_test_ridge,
        df_features_2_step_ahead_ridge,
    ) = feature_engineering(
        df_station=df_station,
        lag=lag_value_ridge,
        q_step_size=2,
        time_sampling_interval_dt=time_sampling_interval_dt,
        short_term_features_flag=True,
        daily_features_flag=True,
        weekly_features_flag=True,
    )

    (
        XX_kNR,
        yy_kNR,
        X_train_kNR,
        y_train_kNR,
        X_test_kNR,
        y_test_kNR,
        df_features_2_step_ahead_kNR,
    ) = feature_engineering(
        df_station=df_station,
        lag=lag_value_kNNR,
        q_step_size=2,
        time_sampling_interval_dt=time_sampling_interval_dt,
        short_term_features_flag=True,
        daily_features_flag=True,
        weekly_features_flag=True,
    )

    # Cross Validation
    PolynomialOrderCrossValidation(XX_ridge, yy_ridge, XX_kNR, yy_kNR)
    polynomial_order_ridge = int(
        input("Please enter the polynomial order 'q' value for Ridge Regression :    ")
    )
    polynomial_order_kNR = int(
        input(
            "Please enter the polynomial order 'q' value for KNeighborsRegressor :    "
        )
    )

    XX_poly_ridge = XX_ridge
    XX_poly_kNR = XX_kNR
    if polynomial_order_ridge > 1:
        XX_poly_ridge = PolynomialFeatures(polynomial_order_ridge).fit_transform(
            XX_poly_ridge
        )
        X_train_ridge = PolynomialFeatures(polynomial_order_ridge).fit_transform(
            X_train_ridge
        )
        X_test_ridge = PolynomialFeatures(polynomial_order_ridge).fit_transform(
            X_test_ridge
        )

    if polynomial_order_kNR > 1:
        XX_poly_kNR = PolynomialFeatures(polynomial_order_kNR).fit_transform(
            XX_poly_kNR
        )
        X_train_kNR = PolynomialFeatures(polynomial_order_ridge).fit_transform(
            X_train_kNR
        )
        X_test_kNR = PolynomialFeatures(polynomial_order_ridge).fit_transform(
            X_test_kNR
        )

    RidgeAlphaValueCrossValidation(XX_poly_ridge, yy_ridge)
    RidgeAlphaValueCrossValidation_method1(XX_poly_ridge, yy_ridge)
    C_value_ridge = float(
        input("Please choose the desired 'C' value for the Ridge Regression model:    ")
    )

    KNeighborsRegressor_k_value_CV(XX_poly_ridge, yy_ridge, 100)
    k_value = int(
        input(
            "Please enter the number of neighnours 'k' value for the KNeighborsRegressor model:    "
        )
    )

    # -----------------------------------------Ridge Regression---------------------------------------
    model_ridgeReg, scores_ridgeReg = ridgeRegression(
        X_train_ridge, y_train_ridge, X_test_ridge, y_test_ridge, C_value_ridge
    )
    ypred_full_ridge = model_ridgeReg.predict(XX_poly_ridge)

    print("Plotting Ridge Regression Predictions")
    plot_preds(
        df_total_station_data.index,
        df_total_station_data["AVAILABLE BIKES"],
        XX_poly_ridge.index,
        ypred_full_ridge,
        station_id,
    )
    print(scores_ridgeReg)

    # -----------------------------------------KNeigborsRegressor---------------------------------------
    model_kNR, scores_kNR = kNearestNeighborsRegression(
        X_train_kNR, y_train_kNR, X_test_kNR, y_test_kNR, k_value
    )
    ypred_full_kNR = model_kNR.predict(XX_poly_kNR)

    print("Plotting Ridge Regression Predictions")
    plot_preds(
        df_total_station_data.index,
        df_total_station_data["AVAILABLE BIKES"],
        XX_poly_kNR.index,
        ypred_full_kNR,
        station_id,
    )
    print(scores_kNR)

    # ----------------------------------------------Baseline-------------------------------------------------
    baselineModel(yy_ridge, station_id, 288)


def main():
    df_dublin_bikes = pd.read_csv(DATASET_PATH)
    df_dublin_bikes["TIME"] = pd.to_datetime(df_dublin_bikes["TIME"])
    df_dublin_bikes.rename(columns={"STATION ID": "STATION_ID"}, inplace=True)
    for station_id in SELECTED_STATIONS:
        df_station = df_dublin_bikes.loc[df_dublin_bikes["STATION_ID"] == station_id]
        exam_2021(df_station, station_id)
        break


if __name__ == "__main__":
    main()
