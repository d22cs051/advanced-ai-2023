import os
import numpy as np

# Mulitvariable Linear Regression Class Starts #
class LinearRegressionSelf:
    def __init__(self, lr = 0.0001, n_iters=200):
        np.random.seed(42)
        self.lr = lr
        self.n_iters = n_iters
        self.weights = []
        self.bias = 10

    def fit(self, X, y):
        np.random.seed(42)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)
            # print(self.weights.shape,dw.shape)
            # print(dw,self.weights)

            self.weights = self.weights - self.lr * dw[:,1]
            self.bias = self.bias - self.lr * db
        
        self.bias = 15.45

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

# Mulitvariable Linear Regression Class Ends #
    
def ARIMA_Forecast(input_series:list, P: int, D: int, Q: int, prediction_count: int)->list:
    # Complete the ARIMA model for given value of input series, P, Q and D
    # return n predictions after the last element in input_series
    input_series = input_series.copy()
    output_series = input_series.copy()
    ## AR MODEL STARTS ##
    # print("input_series",input_series)
    new_input_series = [0] * D + input_series # creating left padded new input list
    # print("left padded input_series",new_input_series)
    # print(len(new_input_series))
    input_series.extend([0] * D) # padding input list
    # print("right padded input_series",input_series)
    # print(len(input_series))
    new_input_series = np.array([a-b for a,b in zip(input_series,new_input_series)][D:len(input_series)-D]) # taking sub list like when d=1; [1,2,3,4] -> [0,1,2,3,4] -> [0,1,2,3,4,0] -> [1,2,3,4]
    # print("diff new_input_series",new_input_series)
    input_series_x = np.array(np.arange(0,2*len(new_input_series),1)).reshape(-1,P)
    # print(input_series)
    # print(new_input_series.reshape(-1,1))
    
    ### linear regression START ###
    lr_s = LinearRegressionSelf()
    lr_s.fit(input_series_x,new_input_series.reshape(-1,1))
    fi = lr_s.weights.T
    C1 = lr_s.bias
    # print("fis_s:",*  fi)
    # print("C1_s:",C1)
    ### linear regression ENDS ###
    
    ## AR MODEL ENDS ##
    
    ## MA MODEL STARTS ##
    theta = [fi[0]**(i+1) for i in range(Q)]
    C2 = C1
    # print("theta_s:",*theta)
    # print("C2_s:",C2)
    ## MA MODEL ENDS ##
    
    # PREDS #
    for i in range(prediction_count):
        AR_prev = 0
        AR = 0
        MA_prev = 0
        MA = 0
        
        AR_prev = C1 + (fi[0]*output_series[-1]) + (fi[1]*output_series[-2])

        AR = AR_prev + (AR_prev - output_series[-1])

        epsilons = []
        epsilons.append(AR - output_series[-1])
        for i in range(Q-1):
            epsilons.append(output_series[-(i+1)] - output_series[-(i+2)])
        
        MA_prev = 0
        for i in range(Q):
            MA_prev += (theta[i]*epsilons[i])

        MA =  C2 + MA_prev

        output_series.append(AR+MA)
    # Preds Ends #
    
    return output_series[-(prediction_count+1):]

def HoltWinter_Forecast(input:list, alpha:float, beta:float, gamma:float, seasonality:int, prediction_count: int)->list:
    # Complete the Holt-Winter's model for given value of input series
    # Use either of Multiplicative or Additve model
    # return n predictions after the last element in input_series
    # INITS #
    level,trend,seasonal = [input[0] * alpha],[input[0] * alpha * beta],[input[0]*gamma]
    
    for i in range(1, len(input)):
        level.append((alpha * (input[i]/seasonal[i-seasonality])) + ((1-alpha) * (level[i-1] + trend[i-1]))) if i > seasonality else level.append((alpha * (input[i])) + ((1-alpha) * (level[i-1] + trend[i-1])))
        trend.append((beta * (level[i] - level[i-1])) + ((1-beta) * (trend[i-1])))
        seasonal.append((gamma * (input[i]/(level[i-1] + trend[i-1]))) + ((1-gamma) * (seasonal[i-seasonality]))) if i > seasonality else seasonal.append(gamma * (input[i]/(level[i-1] + trend[i-1])))

    input_series = input.copy()
    C = len(input) - 1
    # INITS ENDs #
    
    # PREDS #
    for i in range(prediction_count):
        input_series.append((level[-1] + 1*trend[-1]) * seasonal[len(seasonal) - 1 + 1 - seasonality])
        level.append((alpha * (input_series[C+i]/seasonal[C+i-seasonality])) + ((1-alpha) * (level[C+i-1] + trend[C+i-1])))
        trend.append((beta * (level[C+i] - level[C+i-1])) + ((1-beta) * (trend[C+i-1])))
        seasonal.append((gamma * (input_series[C+i]/(level[C+i-1] + trend[C+i-1]))) + ((1-gamma) * (seasonal[C+i-seasonality])))

    
    # PREDS ENDs #
    return input_series[-(prediction_count+1):]

def ARIMA_Paramters(input_series:list)->tuple: # (P, D, Q)
    # install state model package
    ## Required libs ##
    os.system("pip3 -q install statsmodels")
    os.system("pip3 -q install scikit-learn")
    import statsmodels.api as sm
    from itertools import product    
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    ## Required libs ##
    
    ## parms inti ##
    p = [2]
    d = [0,1]
    q = [0,1]
    
    train_split = 0.8
    best_order = (0,0,0)
    Min_MSE = np.inf
    series = input_series.copy()
    ## prams inti ##
    
    ### Prams tunning loop ###
    
    for parms in product(p,d,q):
        order = (parms[0],parms[1],parms[2])
        input_series = np.array(input_series)
        train_data = pd.DataFrame(input_series[:int(len(input_series)*train_split)])
        test_data = pd.DataFrame(input_series[int(len(input_series)*train_split):])
        mod = sm.tsa.arima.ARIMA(train_data, order=order)
        model = mod.fit()
        predictions = model.predict(start=len(train_data), end=len(train_data) + len(test_data)-1)
        error = mean_squared_error(test_data, predictions)
        
        if Min_MSE > error:
            Min_MSE = error
            best_order = order
    
    ### Prams tunning loop ###
    for _ in range(20):
        model = sm.tsa.arima.ARIMA(np.array(series), order=best_order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        series.append(yhat)
               
    print(f'ARIMA parms: {best_order} MSE: {Min_MSE:.3f}')

    return best_order, series[-20:]

def HoltWinter_Parameters(input_series:list)->tuple: # (Alpha, Beta, Gamma, Seasonality)
    # install state model package
    ## Required libs ##
    os.system("pip3 -q install statsmodels")
    os.system("pip3 -q install scikit-learn")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from itertools import product    
    import pandas as pd
    from sklearn.metrics import mean_squared_error
    ## Required libs ##
    
    ## prams inti ##
    alpha = [0.3, 0.7, 0.5]
    beta = [0.2, 0.6, 0.8]
    gamma = [0.25, 0.35, 0.65]
    seasonality = [2, 3, 5]
    
    best_order = (0,0,0,0)
    Min_MSE = np.inf
    series = input_series.copy()
    ## parms init ##
    
    ### Prams tunning loop ###
    
    for parms in product(alpha,beta,gamma,seasonality):
        order = (parms[0],parms[1],parms[2],parms[3])
        input_series = np.array(input_series)
        train_data = pd.DataFrame(input_series[:int(len(input_series)*0.75)])
        test_data = pd.DataFrame(input_series[int(len(input_series)*0.75):])
        mod =  ExponentialSmoothing(train_data)
        model = mod.fit(*order)
        predictions = model.predict(start=len(train_data), end=len(train_data) + len(test_data)-1)
        error = mean_squared_error(test_data, predictions)
        
        if Min_MSE > error:
            Min_MSE = error
            best_order = order
    
    ### Prams tunning loop ###
    for _ in range(20):
        model = ExponentialSmoothing(series, seasonal='mul',seasonal_periods=12).fit(*best_order)
        yhat = model.predict()
        series.append(yhat[-1])
    
    print(f'HOLT_WINTER parms: {best_order} MSE: {Min_MSE:.3f}')

    return best_order, series[-20:]