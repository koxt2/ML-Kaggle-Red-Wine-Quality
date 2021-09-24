import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.impute import SimpleImputer

########## Load CSV data, display headings and check for NULL data ##########

# Load CSV data
def load_red_data():
    return pd.read_csv('/Users/richard/Documents/Python/Machine-Learning/Red-Wine-Quality/red.csv')
red_data = load_red_data()

# Print CSV data and summary
print(red_data.head())
print(red_data.info())
print(red_data.describe())

# Plot histograms
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=200):
    path = os.path.join('/Users/richard/Documents/Python/Machine-Learning/Red-Wine-Quality/', fig_id + "." + fig_extension)
    plt.savefig(path, format=fig_extension, dpi=resolution)

red_data.hist(bins=50, figsize=(15,10))
save_fig("attribute_histogram_plots")

########## Create a test set ##########
# Find which attribute correlates the strongest with quality ( Result: alcohol content)
correlations = red_data.corr()
correlations["quality"].sort_values(ascending=False) 
print(correlations)

# Plot a series of scatter diagrams of other atrributes that show relatively strong correlation
attributes = ["quality", "alcohol", "sulphates",
              "citric acid"]
scatter_matrix(red_data[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")

# To stratify the test set data create an alcohol category atrribute 
red_data["alcohol_cat"] = pd.cut(red_data["alcohol"],
                               bins=[8, 9, 10, 11, 12, 13, np.inf],
                               labels=[8, 9, 10, 11, 12, 13])
print(red_data["alcohol_cat"].value_counts()) # Print the count in each category

# Split from the dataset a stratified sample to use as a test set (20%)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(red_data, red_data["alcohol_cat"]):
    strat_train_set = red_data.loc[train_index]
    strat_test_set = red_data.loc[test_index]

# Check the proportions by alcohol of the test set match the complete dataset
print(strat_test_set["alcohol_cat"].value_counts() / len(strat_test_set))
print(red_data["alcohol_cat"].value_counts() / len(red_data))

# Now the test set and train set are created drop the alcohol category from both
for set_ in (strat_train_set, strat_test_set):
    set_.drop("alcohol_cat", axis=1, inplace=True)

########## Prepare data for ML algorithms ##########
red_data = strat_train_set.drop("quality", axis=1) # Makes a copy of the original data and drops quality (ie, creates predictors)
red_data_labels = strat_train_set["quality"].copy() # Makes a copy and copies quality (ie, creates the labels)

# Use imputer to fill in NULL values
imputer = SimpleImputer(strategy="median")
imputer.fit(red_data)

print("Stats: ", imputer.statistics_)

########## Select a training model ##########
########## Linear Regression ##########
# Apply linear regression to the predictors and labels
#lin_reg = LinearRegression()
#lin_reg.fit(red_data, red_data_labels)

# Run linear regression on some of the data
#some_data = red_data.iloc[:5]
#some_labels = red_data_labels.iloc[:5]

#print("""

#Lin_Reg Predictions:""", lin_reg.predict(some_data))
#print("Labels:", list(some_labels))

# Check the linear regression model on all of the data using Root Mean Square Error
#alcohol_predictions = lin_reg.predict(red_data)
#lin_mse = mean_squared_error(red_data_labels, alcohol_predictions)
#lin_rmse = np.sqrt(lin_mse)
#print("Lin_Reg RMSE: ", lin_rmse)

# Check the linear regression model on all of the data uing Mean Absolute Error
#print("Lin_Reg MAE: ", lin_mae)
#lin_mae = mean_absolute_error(red_data_labels, alcohol_predictions)

# Use cross validation to further evaluate the model
#def display_scores(scores, model):
#    print(model, " Scores:", scores)
#    print(model, " Mean:", scores.mean())
#    print(model, " Standard deviation:", scores.std())

#lin_scores = cross_val_score(lin_reg, red_data, red_data_labels,
#                             scoring="neg_mean_squared_error", cv=10)
#lin_rmse_scores = np.sqrt(-lin_scores)
#display_scores(lin_rmse_scores, "Lin_Reg")

########## Decision Tree Regressor ##########
# Apply decision tree regressor to the predictors and labels
#tree_reg = DecisionTreeRegressor(random_state=42)
#tree_reg.fit(red_data, red_data_labels)

# Run decision tree regressor on some of the data
#alcohol_predictions = tree_reg.predict(red_data)
#tree_mse = mean_squared_error(red_data_labels, alcohol_predictions)
#tree_rmse = np.sqrt(tree_mse)
#print("""

#DTR RMSE""", tree_rmse)

# Use cross validation to further evaluate the model
#scores = cross_val_score(tree_reg, red_data, red_data_labels,
#                         scoring="neg_mean_squared_error", cv=10)
#tree_rmse_scores = np.sqrt(-scores)
#display_scores(tree_rmse_scores, "DTR")

########## Random Forest Regressor ##########
# Apply random tree regressor to the predictors and labels
#forest_reg = RandomForestRegressor(n_estimators=180, random_state=42)
#forest_reg.fit(red_data, red_data_labels)

# Run random forest regressor on some of the data
#alcohol_predictions = forest_reg.predict(red_data)
#forest_mse = mean_squared_error(red_data_labels, alcohol_predictions)
#forest_rmse = np.sqrt(forest_mse)
#print("""

#RFR RMSE""", forest_rmse)

# Use cross validation to further evaluate the model
#scores = cross_val_score(forest_reg, red_data, red_data_labels,
#                         scoring="neg_mean_squared_error", cv=10)
#forest_rmse_scores = np.sqrt(-scores)
#display_scores(forest_rmse_scores, "RFR")

########## Support Vector Regression ##########
# Apply Support Vector Regression to the predictors and labels
#svm_reg = SVR(kernel="linear")
#svm_reg.fit(red_data, red_data_labels)

# Run SVR on some of the data
#alcohol_predictions = svm_reg.predict(red_data)
#svm_mse = mean_squared_error(red_data_labels, alcohol_predictions)
#svm_rmse = np.sqrt(svm_mse)
#print("""

#SVR RMSE""", svm_rmse)

# Use cross validation to further evaluate the model
#scores = cross_val_score(svm_reg, red_data, red_data_labels,
#                         scoring="neg_mean_squared_error", cv=10)
#svm_rmse_scores = np.sqrt(-scores)
#display_scores(svm_rmse_scores, "SVR")

########## Fine Tune Random Forest Regressor using GridSearch ##########
param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [120, 180, 240], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(red_data, red_data_labels)

# Print the results
print("""
#GridSearch Results: """, grid_search.best_estimator_)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

########## FINAL MODEL ##########
final_model = grid_search.best_estimator_

########## Evaluate the final model on the test set ##########
X_test = strat_test_set.drop("quality", axis=1)
y_test = strat_test_set["quality"].copy()

final_predictions = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("""
Final MSE: """, final_mse)
print("Final RMSE: ", final_rmse)

# Calculate the range of a result that has 95% confidence
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
print("95% Confidence Range: ", np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                        loc=squared_errors.mean(),
                        scale=stats.sem(squared_errors))))

########## Use the final model to predict the quality ##########
X_new = [[12.2, None, 0.2, 2.1, 0.081, 14.0, 58.0, 0.999, 3.2, 0.55, 12]] # A new set of data that is going to be used to predict the quality
X_imputed = imputer.transform(X_new) # Transform the imputed value (median) into the new data with NULL(None) value
print("X_imputed: ", X_imputed) # A check to make sure the imputed value was added to the new data
X_new_prediction = final_model.predict(X_imputed) # Perform the prediction
print("Prediction Quality = ", X_new_prediction ) # Print the prediction

# End results are...
# Random Forest Regressor with hyperparameters of (max_features=4, n_estimators=180, random_state=42)
# This provides a final Root Mean Square Error of 0.579, where a 95% confidence range is (0.513 - 0.638)
