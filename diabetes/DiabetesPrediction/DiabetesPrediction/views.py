from django.shortcuts import render

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    dataset = pd.read_csv('/Users/ahmedbinnayeem/Desktop/diabetes/kaggle_diabetes.csv')

    dataset_new = dataset
    del dataset_new["Pregnancies"]
    del dataset_new["DiabetesPedigreeFunction"]
    del dataset_new["SkinThickness"]
    del dataset_new["BloodPressure"]

    # Replacing zero values with NaN
    dataset_new[["Glucose", "Insulin", "BMI"]]= dataset_new[["Glucose", "Insulin", "BMI"]].replace(0, np.NaN) 

    # Replacing NaN with mean values
    dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
    dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
    dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)

    # Feature scaling using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    dataset_scaled = sc.fit_transform(dataset_new)

    dataset_scaled = pd.DataFrame(dataset_scaled)

     #Method:OverSampling
    count_class_0, count_class_1 = dataset_new.Outcome.value_counts()

    #Divide by class
    dataset_new_class_0 = dataset_new[dataset_new['Outcome']==0]
    dataset_new_class_1 = dataset_new[dataset_new['Outcome']==1]

    dataset_new_class_1_over= dataset_new_class_1.sample(count_class_0, replace=True)

    dataset_new_test_over = pd.concat([dataset_new_class_0,dataset_new_class_1_over],axis=0)

    # Splitting X and Y
    X = dataset_new_test_over.drop('Outcome', axis='columns')
    Y = dataset_new_test_over['Outcome']

    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42, stratify = Y )
    Y_test.value_counts()

    # Random forest Algorithm
    from sklearn.ensemble import RandomForestClassifier
    ranfor = RandomForestClassifier(n_estimators = 11, criterion = 'entropy', random_state = 42)
    ranfor.fit(X_train, Y_train)

    # Decision tree Algorithm
    from sklearn.tree import DecisionTreeClassifier
    dectree = DecisionTreeClassifier(criterion = 'entropy', random_state = 42)
    dectree.fit(X_train, Y_train)
    # Y_pred_dectree = dectree.predict(X_test)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])

    prediction = ranfor.predict([[val1, val2, val3, val4]])

    result1 = ""
    if prediction==[1]:
        result1 = "Positive"
    else:
        result1 = "Negative"



    return render(request, 'predict.html', {"result2":result1})
