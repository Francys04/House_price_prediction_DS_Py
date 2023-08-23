import matplotlib.pyplot as plt
import seaborn as sns
from src.processing import house_price_dataframe

'''Positive and negative correlation'''
correlation = house_price_dataframe.corr(numeric_only=True)

'''constructing a heatmap to understand the correlation'''
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, fmt='.1f', annot=True, annot_kws={'size': 10}, cmap='Reds')
plt.show()
