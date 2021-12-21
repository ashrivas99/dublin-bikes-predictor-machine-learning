
from sklearn.neighbors import KNeighborsRegressor

# ___________________________Lag cross validation _________________________________________________________

def lagCrossValidation( y_available_bikes, time_full_days, time_sampling_interval_dt):
    mean_error = []; std_error = []
    lag_range = list(range(1, 10))
    
    test_model_ridge = Ridge(fit_intercept=False); test_model_kNNR = KNeighborsRegressor(n_neighbors=100)
    models = [test_model_ridge, test_model_kNNR]

    step_size = [2,6,12]
    for model in models:
        for q_value in step_size:
            mean_error = []; std_error = []
            for lag_value in lag_range:
                XX_test, yy_test, time_preds_days = feature_engineering(
                    q_step_size= q_value,
                    lag= lag_value,
                    stride=1,
                    y_available_bikes=y_available_bikes,
                    time_full_days=time_full_days,
                    time_sampling_interval_dt=time_sampling_interval_dt,
                    weekly_features_flag=True,
                    daily_features_flag=True,
                    short_term_features_flag=True,
                )
                scores = cross_val_score(model, XX_test, yy_test, cv=10, scoring="neg_mean_squared_error")
                mean_error.append(np.array(scores).mean())
                std_error.append(np.array(scores).std())
            plt.rc("font", size=18)
            plt.rcParams["figure.constrained_layout.use"] = True
            plt.errorbar(lag_range, mean_error, yerr=std_error, linewidth=3)
            plt.xlabel("lag")
            plt.ylabel("negative mean squared error")
            plt.title(f"Lag Cross Validation Results,{q_value*(time_sampling_interval_dt/60)} minutes ahead Preidctions")
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


# ___________________________Training Ridge Regression using k fold cross validation _________________________________________________________
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from Backup.dublin_bikes_predictor import KNeighborsRegressor_k_value_CV, feature_engineering, regression_evaluation_metircs


def ridgeRegression(XX, yy, train, test, C_value, time_preds_days, station_id ):
    ridge_metrics_scores_dict = {}
    temp_mse =[]; temp_rmse =[]; temp_mae =[]; temp_r2 =[]; temp_median_ae =[]

    ridgeRegression_model = Ridge(alpha= (1/(2*C_value)) ,fit_intercept=False)
    kf = KFold(n_splits=50)

    for train_set, test_set in kf.split(XX):
        ridgeRegression_model.fit(XX[train_set], yy[train_set])
        y_pred = ridgeRegression_model.predict(XX[test_set]) 
        ridge_metrics_scores_dict = regression_evaluation_metircs(yy[test_set], y_pred) 
        print(ridgeRegression_model.intercept_, ridgeRegression_model.coef_)
        temp_mse.append(ridge_metrics_scores_dict['mean_sq_err'])
        temp_rmse.append(ridge_metrics_scores_dict['root_mean_sq_err'])
        temp_mae.append(ridge_metrics_scores_dict['mean_abs_err'])
        temp_r2.append(ridge_metrics_scores_dict['r2Score'])
        temp_median_ae.append(ridge_metrics_scores_dict['median_abs_err'])

    mse_overall = np.array(temp_mse).mean()
    rmse_overall = np.array(temp_rmse).mean()
    mae_overall = np.array(temp_mae).mean()
    r2_overall = np.array(temp_r2).mean()
    median_ae_overall = np.array(temp_median_ae).mean()
    
    ridge_metrics_dict = {
        'mean_sq_err': mse_overall ,
        'root_mean_sq_err': rmse_overall ,
        'mean_abs_err': mae_overall ,
        'r2Score': r2_overall,
        'median_abs_err': median_ae_overall
    }

    print(f'Mean Squared Error: {mse_overall} ')
    print(f'Root Mean Squared Error: {rmse_overall}')
    print(f'Mean_Absolute_Error: {mae_overall}')
    print(f'R2 Score: {r2_overall}')
    print(f'Median Absolute Error: {median_ae_overall}')

    print(ridgeRegression_model.intercept_, ridgeRegression_model.coef_)
    ypred = ridgeRegression_model.predict(XX[test])     
    # print('Plotting y_test vs y_predictions')
    # plot_preds(time_preds_days[test], yy[test], time_preds_days[test], ypred, station_id)
    
    return ridgeRegression_model, ridge_metrics_dict

# ____________________________KNN Cross Validation _________________________________________________________
    
def KNeighborsRegressor_k_value_CV_positive_mse(XX, yy):
    mean_error = []
    std_error = []
    num_neighbours = list(range(1, 51))
    for k in num_neighbours:
        kNR_model = KNeighborsRegressor_k_value_CV(n_neighbors=k, weights="uniform")
        temp = []
        kf = KFold(n_splits=10)
        for train, test in kf.split(XX):
            kNR_model.fit(XX[train], yy[train])
            ypred = kNR_model.predict(XX[test])
            temp.append(mean_squared_error(yy[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(num_neighbours, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("k: number of neighbours")
    plt.ylabel("Mean square error")
    plt.title("KNeighborsRegressor Cross Validation Results")
    plt.show()