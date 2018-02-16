#---- fit.py
#       Train and test. Currently doesn't save fit results to disk.
#       If we want to save results, use sklearn.externals.joblib.dump(fit, "FILE_PATH").
#----

#---- function fit_BR
#       Fit Bayesian Ridge Regression results.
#   - Input:
#       Train_X, Train_Y: 
#         Training and validation sets. Need split object to define how to fold.
#       Test_X, Test_Y: 
#         Testing set and its "ground true value" (B3LYP).
#       Predefined_Split: object sklearn.model_selection.PredefinedSplit
#         Cross validation fold definition file. Obtained by function fold.predefined_fold()
#   - Output: (tuple)
#       Err_MAD: float
#         Mean absolute deviation of prediction of testing set.
#       Err_RMSD: float
#         Root mean squared deviation of prediction of testing set.
#       Time_Train: float
#         The time of training.
#       Time_Test: float
#         The time of testing.
#   - Variable:
#       Fit_BR: object sklearn.model_selection.GridSearchCV
#         Fit model of Bayesian Regression.
#       Time0, Time1: time float
#       Predict_Y: np.array
#         Prediction value of Test_X.
#----
def fit_BR(Train_X, Train_Y, Test_X, Test_Y, Predefined_Split):
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import BayesianRidge
    import time
    #  1. Fitting
    # Probably there is no parameter grid search.
    Fit_BR = GridSearchCV(\
        BayesianRidge(), cv = Predefined_Split, \
        param_grid = {})
    Time0 = time.time()
    Fit_BR.fit(Train_X, Train_Y)
    Time1 = time.time()
    Time_Train = Time1 - Time0
    #  2. Prediction and Error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    Time0 = time.time()
    Predict_Y = Fit_BR.predict(Test_X)
    Err_MAD  = mean_absolute_error(Test_Y, Predict_Y)
    Err_RMSD = mean_squared_error (Test_Y, Predict_Y)
    Time1 = time.time()
    Time_Test = Time1 - Time0
    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test)

#---- function fit_EN
#       Fit Elastic Net Regression results.
#       Almost all the variables in the function are the same as fit_BR.
#----
def fit_EN(Train_X, Train_Y, Test_X, Test_Y, Predefined_Split):
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import ElasticNet
    import numpy as np
    import time
    #  1. Fitting
    # In article, 
    # - l1_ratio: 0.5
    # - alpha: log: 1e-6 -> 1e0, base 10
    Fit_EN = GridSearchCV(\
        ElasticNet(l1_ratio = 0.5), cv = Predefined_Split, \
        param_grid = {"alpha": np.logspace(-6,0,7)})
    Time0 = time.time()
    Fit_EN.fit(Train_X, Train_Y)
    Time1 = time.time()
    Time_Train = Time1 - Time0
    #  2. Prediction and Error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    Time0 = time.time()
    Predict_Y = Fit_EN.predict(Test_X)
    Err_MAD  = mean_absolute_error(Test_Y, Predict_Y)
    Err_RMSD = mean_squared_error (Test_Y, Predict_Y)
    Time1 = time.time()
    Time_Test = Time1 - Time0
    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test)

#---- function fit_RF
#       Fit Random Forest Regression results.
#       Almost all the variables in the function are the same as fit_BR.
#----
def fit_RF(Train_X, Train_Y, Test_X, Test_Y, Predefined_Split):
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    import time
    #  1. Fitting
    # In article, 
    # - n_estimators: 120
    #     Reported number of trees is defined by 120.
    #   All other hyperparameters are not explored. Default values are chosen here.
    Fit_RF = GridSearchCV(\
        RandomForestRegressor(n_estimators = 120, n_jobs=-1), cv = Predefined_Split, \
        param_grid = {})
    Time0 = time.time()
    Fit_RF.fit(Train_X, Train_Y)
    Time1 = time.time()
    Time_Train = Time1 - Time0
    #  2. Prediction and Error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    Time0 = time.time()
    Predict_Y = Fit_RF.predict(Test_X)
    Err_MAD  = mean_absolute_error(Test_Y, Predict_Y)
    Err_RMSD = mean_squared_error (Test_Y, Predict_Y)
    Time1 = time.time()
    Time_Test = Time1 - Time0
    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test)

#---- function fit_KRR
#       Fit Kernel Ridge Regression results.
#       Almost all the variables in the function are the same as fit_BR.
#----
def fit_KRR(Train_X, Train_Y, Test_X, Test_Y, Predefined_Split):
    from sklearn.model_selection import GridSearchCV
    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    import time
    #  1. Fitting
    # In article, 
    # - alpha: 1e-9
    #     alpha is probably the penalty of ridge regularization parameter.
    # - kernel: "RBF", "laplacian"
    # - gamma: log: 0.125 -> 16384, base: 2
    #     gamma is probably the width of kernel.
    #     Same serach gird are set to both RBF and Laplacian kernels.
    # Feature vector normalization is not employed.
    Fit_KRR = GridSearchCV(\
        KernelRidge(alpha=1e-9), cv = Predefined_Split, \
        param_grid = {"kernel": ["rbf", "laplacian"], \
                      "gamma": np.logspace(-3,14, base=2., num=18)})
    Time0 = time.time()
    Fit_KRR.fit(Train_X, Train_Y)
    Time1 = time.time()
    Time_Train = Time1 - Time0
    #  2. Prediction and Error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    Time0 = time.time()
    Predict_Y = Fit_KRR.predict(Test_X)
    Err_MAD  = mean_absolute_error(Test_Y, Predict_Y)
    Err_RMSD = mean_squared_error (Test_Y, Predict_Y)
    Time1 = time.time()
    Time_Test = Time1 - Time0
    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test)

#---- function fit_KRR_multi
#       Fit Kernel Ridge Regression results.
#       Multi-variate regression is employed.
#       Different shape of Y, Err is shown here.
#----
def fit_KRR_multi(Train_X, Train_Y, Test_X, Test_Y, Predefined_Split):
    from sklearn.model_selection import GridSearchCV
    from sklearn.kernel_ridge import KernelRidge
    import numpy as np
    import time
    #  1. Fitting
    # In article, 
    # - alpha: np.ones(13)*1e-9
    #     alpha is probably the penalty of ridge regularization parameter.
    #     Parameter for all the targets are chosen to be equal to each other.
    # - kernel: "RBF", "laplacian"
    # - gamma: log: 0.125 -> 16384, base: 2
    #     gamma is probably the width of kernel.
    #     Same serach gird are set to both RBF and Laplacian kernels.
    # Feature vector normalization is not employed.
    Fit_KRR = GridSearchCV(\
        KernelRidge(alpha=1e-9), cv = Predefined_Split, \
        param_grid = {"kernel": ["rbf", "laplacian"], \
                      "gamma": np.logspace(-3,14, base=2., num=18)})
    Time0 = time.time()
    # The input Train_Y matrix should be transposed to [*, 13].
    Fit_KRR.fit(Train_X, np.transpose(Train_Y))
    Time1 = time.time()
    Time_Train = Time1 - Time0
    #  2. Prediction and Error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    Time0 = time.time()
    Predict_Y = Fit_KRR.predict(Test_X)
    Test_Y_Tr = np.transpose(Test_Y_Tr)
    Err_MAD  = [ mean_absolute_error(Test_Y[:,Indi], Predict_Y[:,Indi]) for Indi in range(13) ]
    Err_RMSD = [ mean_squared_error (Test_Y[:,Indi], Predict_Y[:,Indi]) for Indi in range(13) ]
    Time1 = time.time()
    Time_Test = Time1 - Time0
    #  3. Final return
    return (Err_MAD, Err_RMSD, Time_Train, Time_Test)

