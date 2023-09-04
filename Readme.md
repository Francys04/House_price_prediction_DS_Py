## California Housing Price Prediction Readme
### Overview
 - This repository contains code for exploring, analyzing, and building a predictive model for California housing prices. It uses the California housing dataset available in scikit-learn.

### Code Structure
- Data Loading and Exploration: The code begins by loading the California housing dataset, converting it into a Pandas DataFrame, and performing initial data exploration. This includes checking for missing values, calculating statistical measures, and visualizing correlations between features.

- Data Splitting and Model Training: The dataset is split into training and testing subsets using the train_test_split function from scikit-learn. An XGBoost Regressor model is loaded and trained on the training data.

- Evaluation: The code computes predictions on the training data and evaluates the model's performance using metrics like R-squared and Mean Absolute Error (MAE).

- Data Visualization: Visualization is an essential part of data analysis. The code creates a heatmap to visualize feature correlations and a scatter plot to show the relationship between actual and predicted housing prices.

### Usage
- To run the code, ensure you have the required libraries installed, including scikit-learn, XGBoost, NumPy, Pandas, Matplotlib, and Seaborn.

- Execute the code step by step or as a script to load, explore, and train the model on the California housing dataset.

- Adjust the code as needed for your specific use case. For instance, you can experiment with different machine learning algorithms or hyperparameters to improve model performance.

### Results
- The code provides insights into the California housing dataset, including feature correlations and a predictive model's performance metrics.

- It visualizes the data to help users understand the relationships between different features and the quality of the predictive model.



#### Fig.1 Actual Price vs Predicted Price
<img src="img/actual price.JPG">