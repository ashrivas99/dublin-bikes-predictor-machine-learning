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
from sklearn.model_selection import cross_val_score, train_test_split
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

def feature_engineering():
    pass

# Shift the number of available bikes backwards so that the outputs align with the inputs


def exam_2021(df_station, station_id):

    start_date=pd.to_datetime("29-01-2020",format='%d-%m-%Y')
    df_station = df_station[df_station.TIME>start_date]

    time_full_seconds = (
        pd.array(pd.DatetimeIndex(df_station.iloc[:, 1]).astype(np.int64)) / 1000000000
    )
    time_full_seconds = time_full_seconds.to_numpy()
    time_sampling_interval_dt = time_full_seconds[1] - time_full_seconds[0]
    print(
        f"data sampling interval is {time_sampling_interval_dt} secs or {time_sampling_interval_dt/60} minutes"
    )

    step_size_10_min_ahead_preds = 2; step_size_30_min_ahead_preds = 6 ; step_size_1_hr_ahead_preds = 12

    df_station = df_station.set_index('TIME')
    available_bikes_df = df_station[['AVAILABLE BIKES']]
    available_bikes_df = available_bikes_df.dropna()


    
    available_bikes_df1 = df_station[['AVAILABLE BIKES']]
    available_bikes_df1 = available_bikes_df1.dropna()

    available_bikes_df.loc[:,'bikes_avail_10_mins_ahead'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(-step_size_10_min_ahead_preds)
    available_bikes_df.loc[:,'bikes_avail_30_mins_ahead'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(-step_size_30_min_ahead_preds)
    available_bikes_df.loc[:,'bikes_avail_30_mins_ahead'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(-step_size_1_hr_ahead_preds)
    

    available_bikes_df.loc[:,'1_point_before'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift()
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'2_point_before'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(2)
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'3_point_before'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(3)
    available_bikes_df = available_bikes_df.dropna()

    # available_bikes_df = available_bikes_df1.copy()
    available_bikes_df.loc[:,'Bikes_Yesterday'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(287)
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df['Bikes_2_days_ago'] = available_bikes_df['AVAILABLE BIKES'].shift(288)
    available_bikes_df = available_bikes_df.dropna()
    
    available_bikes_df['Bikes_3_days_ago'] = available_bikes_df['AVAILABLE BIKES'].shift(289)
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'1_week_ago'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(2015)
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'2_week_ago'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(2015)
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'3_week_ago'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(2015)
    available_bikes_df = available_bikes_df.dropna()

    plot_station_data(available_bikes_df.index, available_bikes_df['AVAILABLE BIKES'],19)

    train_indices = 0.70*available_bikes_df.shape[0]

    XX = available_bikes_df.drop(['AVAILABLE BIKES'], axis = 1)
    X_train = available_bikes_df[:int(train_indices)].drop(['AVAILABLE BIKES'], axis = 1)
    X_test = available_bikes_df[int(train_indices):].drop(['AVAILABLE BIKES'], axis = 1)
    y_train = available_bikes_df[:int(train_indices)]; y_train = y_train[['AVAILABLE BIKES']]
    y_test = available_bikes_df[int(train_indices):]; y_test = y_test[['AVAILABLE BIKES']]

    from sklearn.model_selection import TimeSeriesSplit
    # Spot Check Algorithms
    models = []
    models.append(('LR', Ridge(fit_intercept =True)))
    models.append(('KNN', KNeighborsRegressor())) 
    models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        # TimeSeries Cross validation
        tscv = TimeSeriesSplit(n_splits=10)
        cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
        
    # Compare Algorithms
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.show()

    from sklearn.metrics import make_scorer
    def rmse(actual, predict):
        predict = np.array(predict)
        actual = np.array(actual)
        distance = predict - actual
        square_distance = distance ** 2
        mean_square_distance = square_distance.mean()
        score = np.sqrt(mean_square_distance)
        return score
       
    rmse_score = make_scorer(rmse, greater_is_better = False)
    from sklearn.model_selection import GridSearchCV
    model = RandomForestRegressor()
    param_search = { 
        'n_estimators': [20, 50, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [i for i in range(5,15)]
    }
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
    gsearch.fit(X_train, y_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    y_true = y_test.values
    y_pred = best_model.predict(X_test)
    regression_evaluation_metircs(y_true, y_pred)
    plot_preds(available_bikes_df.index, available_bikes_df['AVAILABLE BIKES'],X_test.index, y_test,19)

# _____
    
    model = Ridge()
    param_search = { 
       'alpha':[1, 10]
    }
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
    gsearch.fit(X_train, y_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    y_true = y_test.values
    y_pred = best_model.predict(X_test)
    regression_evaluation_metircs(y_true, y_pred)
    plot_preds(available_bikes_df1.index, available_bikes_df1['AVAILABLE BIKES'], XX.index, best_model.predict(XX),19)

    reg = RidgeCV()
    reg.fit(X_train,y_train)
    print("Best alpha using built-in RidgeCV: %f" % reg.alpha_)
    print("Best score using built-in RidgeCV: %f" %reg.score(X_train,y_train))
    weights = reg.coef_.reshape(9)
    coef = pd.Series(weights, index = X_train.columns.to_numpy())
    print("Ridge picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    imp_coef = coef.sort_values()
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind = "barh")
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


# Idea -> weekday -> week -> day of the week-> time