
# ___________________________Training Ridge Regression using k fold cross validation _________________________________________________________
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from dublin_bikes_predictor import KNeighborsRegressor_k_value_CV, regression_evaluation_metircs


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