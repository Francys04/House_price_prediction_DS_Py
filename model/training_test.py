from sklearn.model_selection import train_test_split
from src.processing import X, Y
from xgboost import XGBRegressor
from sklearn import metrics

'''splitting  the data intro training data and test data'''

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

'''Model training'''
'''XGBoost Regressor'''

# loading the model
model = XGBRegressor()

'''training the model with X_train'''
print(model.fit(X_train, Y_train))

'''Evaluation'''
'''prediction on training'''
# accuracy for prediction on training data
training_data_prediction = model.predict(X_train)
print(training_data_prediction)

# # R squared error
# score_1 = metrics.r2_score(Y_train, training_data_prediction)
#
# # Mean absolut error
# score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
#
# print("R square error : ", score_1)
# print("Mean Absolute error : ", score_2)

# '''accuaracy for prediction on test data'''
# test_data_prediction = model.predict(X_test)
#
# # R squared error
# score_1 = metrics.r2_score(Y_test, test_data_prediction)
#
# # Mean absolut error
# score_2 = metrics.mean_absolute_error(X_test, test_data_prediction)
#
# print("R square error : ", score_1)
# print("Mean Absolute error : ", score_2)
