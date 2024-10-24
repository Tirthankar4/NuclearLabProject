import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from Peak_finder import *
    
#X2 = pd.read_csv(f"C:/Users/tirth/OneDrive/Desktop/Nuclear_project/CSV files/CL_07_E02.csv")
#X3 = pd.read_csv(f"C:/Users/tirth/OneDrive/Desktop/Nuclear_project/CSV files/CL_07_E03.csv")
#X4 = pd.read_csv(f"C:/Users/tirth/OneDrive/Desktop/Nuclear_project/CSV files/CL_07_E04.csv")

Map = {251.5: 81.5, 378.5: 121.782, 760.5: 244.697, 859.5: 276.5, 941.5: 303.5, 1069.5: 344.278, 1106.5: 356.5, 1192.5: 384.5,
       1277.5: 411.5, 1378.5: 444.5, 2420.5: 778.904, 2695.5: 867.380, 2996.5: 964.057, 3372.5: 1086.5, 3455.5: 1112.076,
       4374.5: 1408.013}

X_train = pd.DataFrame(Map.keys())
y_train = pd.Series(Map.values()).reindex(X_train.index)
x_train = list(Map.keys())

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#linear_slope = np.polyfit(x_train, y_train, 1)

def linear_fit(df, n = 10, max_allowed_error = 0.01):
    
    parts = split_dataframe(df, n)
    train_test = find_peaks(parts, max_allowed_error)
    x_train = np.array(list(train_test.keys())).reshape(-1, 1)
    x_test = np.array(list(train_test.values()))#.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x_train, x_test)

    #linear_slope = np.polyfit(x_train, x_test, 1)

    predict = model.predict(x_train).flatten()

    errors = []

    for i in range(len(x_train)):
        errors.append(round((x_test[i] - predict[i]), 3))

    return linear_slope, x_train.flatten(), x_test, errors#, predict,

def poly_fit(df, degree, n = 10, max_allowed_error = 0.01):

    poly = PolynomialFeatures(degree = degree)

    parts = split_dataframe(df, n)
    train_test = find_peaks(parts, max_allowed_error)
    x_train = np.array(list(train_test.keys())).reshape(-1, 1)
    x_test = np.array(list(train_test.values()))#.reshape(-1, 1)

    #slope = np.polyfit(x_train, x_test, 2)

    x_train_poly = poly.fit_transform(x_train)

    model = LinearRegression()
    model.fit(x_train_poly, x_test)

    predict = model.predict(x_train_poly).flatten()

    errors = []

    for i in range(len(x_train)):
        #value = slope[2] + slope[1]*x_train[i] + slope[0]*(x_train[i])**2
        errors.append(round((x_test[i] - predict[i]), 3))

    return model.coef_, x_train.flatten(), x_test, errors

'''def poly_file_fit(df, degree, max_allowed_error = 0.5, Ba_prom = 1, Eu_prom = 1):

    train = peak_find(df, 1, 1)

    model, poly = poly_fit(df, Ba_prom, Eu_prom, degree)

    train_poly = poly.fit_transform(train)
    preds = model.predict(train_poly)
    
    final_train = []

    for i in range(len(train)):
        clst_num = find_closest_number(list(y_train), preds[i])
        
        if abs(clst_num - preds[i]) < max_allowed_error:
            final_train.append(train[i])

    final_train_poly = poly.transform(final_train)

    predict = model.predict(final_train_poly).flatten()
    for i in range(len(predict)):
        predict[i] = round(predict[i], 3)
        
    test = []
    errors = []

    for i in range(len(predict)):
        clst_num = find_closest_number(list(y_train), predict[i])
        test.append(round((clst_num), 2))
        errors.append(round((clst_num - predict[i]), 3))

    dataframe = pd.DataFrame({'train': final_train, 'predict': predict, 'actual': test, 'errors': errors})
    
    min_error_indices = dataframe.groupby('actual')['errors'].idxmin()

    filtered_train = dataframe.loc[min_error_indices, 'train'].tolist()
    filtered_predict = dataframe.loc[min_error_indices, 'predict'].tolist()
    filtered_test = dataframe.loc[min_error_indices, 'actual'].tolist()
    filtered_errors = dataframe.loc[min_error_indices, 'errors'].tolist()
    
    return model.coef_, np.concatenate(filtered_train), filtered_predict, filtered_test, filtered_errors'''

    
