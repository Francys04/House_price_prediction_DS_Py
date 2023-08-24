import matplotlib.pyplot as plt
import seaborn as sns
from src.processing import house_price_dataframe
from model.training_test import Y_train, training_data_prediction

'''Positive and negative correlation'''
correlation = house_price_dataframe.corr(numeric_only=True)

'''constructing a heatmap to understand the correlation'''
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, fmt='.1f', annot=True, annot_kws={'size': 10}, cmap='Reds')
plt.show()


'''visualizing the actual price and predicted prices'''
plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Price vs Predicted Price")
plt.show()
