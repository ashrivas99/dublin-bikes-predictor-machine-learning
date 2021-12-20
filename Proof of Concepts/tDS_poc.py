import math
from operator import index
from matplotlib.style import available

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
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

import sklearn.metrics as metrics
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))


def exam_2021(df_station, station_id):

    start_date=pd.to_datetime("29-01-2020",format='%d-%m-%Y')
    df_station = df_station[df_station.TIME>start_date]

    df_station = df_station.set_index('TIME')
    available_bikes_df = df_station[['AVAILABLE BIKES']]
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'1_point_before'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift()
    available_bikes_df.loc[:,'1_point_before_Diff'] = available_bikes_df.loc[:,'1_point_before'].diff()
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'2_point_before'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(2)
    available_bikes_df.loc[:,'2_point_before_Diff'] = available_bikes_df.loc[:,'2_point_before'].diff()
    available_bikes_df = available_bikes_df.dropna()

    available_bikes_df.loc[:,'3_point_before'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift(3)
    available_bikes_df.loc[:,'3_point_before_Diff'] = available_bikes_df.loc[:,'2_point_before'].diff()
    available_bikes_df = available_bikes_df.dropna()

    # inserting new column with yesterday's consumption values
    available_bikes_df.loc[:,'Bikes_Yesterday'] = available_bikes_df.loc[:,'AVAILABLE BIKES'].shift()
    # inserting another column with difference between yesterday and day before yesterday's consumption values.
    available_bikes_df.loc[:,'Bikes_Yesterday_Diff'] = available_bikes_df.loc[:,'Bikes_Yesterday'].diff()
    available_bikes_df = available_bikes_df.dropna()
    plot_station_data(available_bikes_df.index, available_bikes_df['AVAILABLE BIKES'],19)

    X_train = available_bikes_df[:'2020-02'].drop(['AVAILABLE BIKES'], axis = 1)
    y_train = available_bikes_df.loc[:'2020-02', 'AVAILABLE BIKES']
    X_test = available_bikes_df['2020-03'].drop(['AVAILABLE BIKES'], axis = 1)
    y_test = available_bikes_df.loc['2020-03', 'AVAILABLE BIKES']

    from sklearn.model_selection import TimeSeriesSplit
    # Spot Check Algorithms
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('NN', MLPRegressor(solver = 'lbfgs')))  #neural network
    models.append(('KNN', KNeighborsRegressor())) 
    models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees
    models.append(('SVR', SVR(gamma='auto'))) # kernel = linear
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
    regression_results(y_true, y_pred)
    plot_preds(available_bikes_df.index, available_bikes_df['AVAILABLE BIKES'],X_test.index, y_test,19)

    # creating copy of original dataframe
    available_bikes_df_2o = available_bikes_df.copy()
    # inserting column with yesterday-1 values
    available_bikes_df_2o['Bikes_Yesterday-1'] = available_bikes_df_2o['Bikes_Yesterday'].shift(288)
    # inserting column with difference in yesterday-1 and yesterday-2 values.
    available_bikes_df_2o['Bikes_Yesterday-1_Diff'] = available_bikes_df_2o['Bikes_Yesterday-1'].diff()
    # dropping NAs
    available_bikes_df_2o = available_bikes_df_2o.dropna()

    X_train_2o = available_bikes_df_2o[:'2020-03-11'].drop(['AVAILABLE BIKES'], axis = 1)
    y_train_2o = available_bikes_df_2o.loc[:'2020-03-11', 'AVAILABLE BIKES']
    X_test_2o = available_bikes_df_2o['2020-03-12':].drop(['AVAILABLE BIKES'], axis = 1)
    y_test_2o = available_bikes_df_2o.loc['2020-03-12':, 'AVAILABLE BIKES']
    
    model = Ridge()
    param_search = { 
       'alpha':[1, 10]
    }
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
    gsearch.fit(X_train_2o, y_train_2o)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    y_true = y_test_2o.values
    y_pred = best_model.predict(X_test_2o)
    regression_results(y_true, y_pred)
    plot_preds(available_bikes_df.index, available_bikes_df['AVAILABLE BIKES'],X_test_2o.index, y_test_2o,19)

    imp = best_model.feature_importances_
    features = X_train_2o.columns
    indices = np.argsort(imp)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), imp[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

    available_bikes_df_weekly = available_bikes_df_2o.copy() 
    available_bikes_df_weekly['Last_Week'] = available_bikes_df_weekly['AVAILABLE BIKES'].shift(2016)
    available_bikes_df_weekly = available_bikes_df_weekly.dropna()

    X_train_weekly = available_bikes_df_weekly[:'2020-01'].drop(['AVAILABLE BIKES'], axis = 1)
    y_train_weekly = available_bikes_df_weekly.loc[:'2020-01', 'AVAILABLE BIKES']
    X_test_weekly = available_bikes_df_weekly['2020-02':].drop(['AVAILABLE BIKES'], axis = 1)
    y_test_weekly = available_bikes_df_weekly.loc['2020-02':, 'AVAILABLE BIKES']
    
    model = RandomForestRegressor()
    param_search = { 
        'n_estimators': [20, 50, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [i for i in range(5,15)]
    }
    # model = Ridge()
    # param_search = { 
    #    'alpha':[1, 10]
    # }
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = rmse_score)
    gsearch.fit(X_train_weekly, y_train_weekly)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_
    y_true = y_test_weekly.values
    y_pred = best_model.predict(X_test_weekly)
    regression_results(y_true, y_pred)
    plot_preds(available_bikes_df.index, available_bikes_df['AVAILABLE BIKES'],X_test_weekly.index, y_test_weekly,19)

    imp = best_model.feature_importances_
    features = X_train_weekly.columns
    indices = np.argsort(imp)
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), imp[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
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
