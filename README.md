This project is focused on predicting rainfall based on features such as cloud cover, humidity, wind speed, temperature, and other weather-related variables. The steps in the workflow include:

Data Preprocessing: Handle missing values, encoding categorical variables, and removing outliers.
Feature Selection: Drop features that have low correlation with the target variable (rainfall).
Balancing Data: Use Random OverSampling to balance the dataset by oversampling the minority class.
Model Training: Train a Logistic Regression model on the resampled and normalized dataset.
Model Evaluation: Evaluate model performance using ROC AUC.
