import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "Data/dublinbikes_20200101_20200401.csv"
       

def main():
    df_dublin_bikes = pd.read_csv(DATASET_PATH)
    df_dublin_bikes["TIME"] = pd.to_datetime(df_dublin_bikes["TIME"])
    df_dublin_bikes.rename(columns={"STATION ID": "STATION_ID"}, inplace=True)
    station = 19
    station_id = df_dublin_bikes.STATION_ID == station
    df_station = df_dublin_bikes[station_id]

    end_date = pd.to_datetime("14-03-2020",format='%d-%m-%Y')

    # convert date/time to unix timestamp in sec
    time_full = pd.array(pd.DatetimeIndex(df_station.iloc[:,1]).astype(np.int64))/1000000000
    time_full = time_full.to_numpy()
    data_time_sampling_interval_dt = time_full[1]-time_full[0]
    print(f"data sampling interval is {data_time_sampling_interval_dt} secs")


    time_full_in_days = (time_full-time_full[0])/60/60/24 # convert timestamp to days
    y_available_bikes = np.extract([time_full], df_station.iloc[:,6]).astype(np.int64)
    # plot extracted data
    plt.scatter(time_full_in_days,y_available_bikes, color='red', marker='.'); plt.show()

    def test_preds(q,dd,lag,plot):
        #q-step ahead prediction
        stride=1
        XX_input_features=y_available_bikes[0:y_available_bikes.size-q-lag*dd:stride]
        
        for i in range(1,lag):
            X=y_available_bikes[i*dd:y_available_bikes.size-q-(lag-i)*dd:stride]
            XX_input_features=np.column_stack((XX_input_features,X))
        
        yy_outputs_available_bikes=y_available_bikes[lag*dd+q::stride]
        time_for_predictions_in_days=time_full_in_days[lag*dd+q::stride]
        
        train, test = train_test_split(np.arange(0,yy_outputs_available_bikes.size),test_size=0.2)
        from sklearn.linear_model import Ridge
        model = Ridge(fit_intercept=False).fit(XX_input_features[train], yy_outputs_available_bikes[train])
        print(model.intercept_, model.coef_)
        
        if plot:
            y_pred = model.predict(XX_input_features)
            plt.scatter(time_full_in_days, y_available_bikes, color='black'); plt.scatter(time_for_predictions_in_days, y_pred, color='blue')
            plt.xlabel("time (days)"); plt.ylabel("#bikes")
            plt.legend(["training data","predictions"],loc='upper right')
            day=math.floor(24*60*60/data_time_sampling_interval_dt) # number of samples per day
            # plt.xlim(((lag*dd+q)/day,(lag*dd+q)/day+2))
            plt.show()
    # prediction using short-term trend
    plot=True
    test_preds(q=10,dd=1,lag=3,plot=plot)
    # prediction using daily seasonality
    d=math.floor(24*60*60/data_time_sampling_interval_dt) # number of samples per day
    test_preds(q=d,dd=d,lag=3,plot=plot)
    # prediction using weekly seasonality
    w=math.floor(7*24*60*60/data_time_sampling_interval_dt) # number of samples per day
    test_preds(q=w,dd=w,lag=3,plot=plot)

    #putting it together
    q=10
    lag=3; stride=1
    w=math.floor(7*24*60*60/data_time_sampling_interval_dt) # number of samples per week
    len = y_available_bikes.size-w-lag*w-q
    XX_input_features = y_available_bikes[q:q+len:stride]
    for i in range(1,lag):
        X = y_available_bikes[i*w+q:i*w+q+len:stride]
        XX_input_features = np.column_stack((XX_input_features,X))
    d=math.floor(24*60*60/data_time_sampling_interval_dt) # number of samples per day
    for i in range(0,lag):
        X = y_available_bikes[i*d+q:i*d+q+len:stride]
        XX_input_features = np.column_stack((XX_input_features,X))
    for i in range(0,lag):
        X = y_available_bikes[i:i+len:stride]
        XX_input_features = np.column_stack((XX_input_features,X))
    yy_outputs_available_bikes = y_available_bikes[lag*w+w+q:lag*w+w+q+len:stride]
    time_for_predictions_in_days = time_full_in_days[lag*w+w+q:lag*w+w+q+len:stride]
  
    train, test = train_test_split(np.arange(0,yy_outputs_available_bikes.size),test_size=0.2)
    from sklearn.linear_model import Ridge
    model = Ridge(fit_intercept=False).fit(XX_input_features[train], yy_outputs_available_bikes[train])
    print(model.intercept_, model.coef_)
    if plot:
        y_pred = model.predict(XX_input_features)
        plt.scatter(time_full_in_days, y_available_bikes, color='black')
        plt.scatter(time_for_predictions_in_days, y_pred, color='blue')
        plt.xlabel("time (days)"); plt.ylabel("#bikes")
        plt.legend(["training data","predictions"],loc='upper right')
        plt.title(f"Available bikes vs predicted available bikes using all")
        day=math.floor(24*60*60/data_time_sampling_interval_dt) # number of samples per day
        # plt.xlim((4*7,4*7+4))
        plt.show()



if __name__ == '__main__':
    main()
