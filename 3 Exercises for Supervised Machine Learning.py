import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

"""1. Regularization"""

#(a) Read the data from yesterday’s exercise ”Feature Engineering” (./output/polynomials.csv) into a pandas.DataFra(b)
# Use column ”y” as target variable and all other columns as predicting variables (named X in class) and
#split them as usual.

df = pd.read_csv("./output/polynomials.csv", sep=",", index_col=0)

#(b) Use column ”y” as target variable and all other columns as predicting variables (named X in class)
#split them as usual.

X = df.loc[:, df.columns != "y"]
y = df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#(c) Learn an ordinary OLS model, a Ridge model and a Lasso model using the provided data with penalty
#parameter equal to 0.3. Using the the R2 scores, which model yields the best prediction?

"""OLS"""
lm = OLS().fit(X_train, y_train)
lm.score(X_test, y_test) #R squared= 0.6686542600757108

"""Ridge"""
# Using Ridge regularization.
ridge = Ridge(alpha=0.3).fit(X_train, y_train)
ridge.score(X_test, y_test) #Score = 0.6625287077872541

"""Lasso"""
# Using Lasso, same alpha
lasso = Lasso(alpha=0.3).fit(X_train, y_train)
lasso.score(X_test, y_test) #Score = 0.6330615954699439


#(d) Create a new pandas.DataFrame() containing the learned coefficients of all
# models and the feature names #as index. In how many rows are the Lasso
# coefficients equal to 0 while the Ridge coefficients are not?

df = pd.DataFrame(lm.coef_, index=X.columns, columns=["OLS"])
df["Lasso"] = lasso.coef_
df["Ridge"] = ridge.coef_

len(df[(df["Lasso"] == 0) & (df["Ridge"] != 0)])
# In 16 rows are the Lasso coefficients equal to 0 while the Ridge coefficients
# are not.


#(e) Using matplotlib.pyplot, create a horizontal bar plot of dimension 10x30
# showing the coefficient sizes.#Save the figure as ./output/polynomials.pdf.

df.plot.barh(figsize=(10, 30))
plt.savefig("./output/polynomials.pdf")

"""2. Neural Network Regression"""

#(a) Load the diabetes dataset using sklearn.datasets.load_diabetes().
# The data is on health and diabetes #of 442 patients. Split the data as usual.

diabetes = load_diabetes()
X = diabetes["data"]
y = diabetes["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#(b) Learn a Neural Network Regressor with identity-activation after
# Standard-Scaling with in total nine #parameter combinations of your choice.
# Use the best solver for weight optimization for this dataset #according to
# the documentation! To keep computational burden low you may use a
# 3-fold Cross-validation and at most 1,000 iterations.

scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

algorithms = [("scaler", MinMaxScaler()),
              ("nn", MLPRegressor(solver="lbfgs", random_state=42,
                                  max_iter=1000))]
pipe = Pipeline(algorithms, verbose=True)
param_grid = {"nn__hidden_layer_sizes": [(75, 75), (90, 90), (100, 100)],
              "nn__activation": ["tanh"],
              "nn__alpha": [0.01, 0.001, 0.005]}

grid = GridSearchCV(pipe, param_grid, return_train_score=True, cv=3)
grid.fit(X, y)

results = pd.DataFrame(grid.cv_results_)

#(c) What are your best parameters? How well do they perform in the training
# set? How well does your model #generalize?

print(grid.best_params_)
print(grid.best_score_)

# The best parameters are layers sizes (100, 100) with alpha .001
# It has a performance of 0.3559 which is not very good.

#(d) Plot a heatmap for the first coefficients matrix of the best model
# ( Access the model via .best_estimator_. #One of its attributes
# is _final_estimator, which behaves like a normal model object.). Be sure
# to label #the correct axis with the feature names. Save the heatmap
# as ./output/nn_diabetes_importances.pdf.

grid.best_estimator_

mlp = MLPRegressor(random_state=42, solver="lbfgs",
                   hidden_layer_sizes=(100, 100),  max_iter=1000,
                   activation="tanh", alpha=0.001)
mlp.fit(X_train_scaled, y_train)
mlp.score(X_test_scaled, y_test)
#Score: 0.4437341781744254

scores = results["mean_test_score"].values.reshape(3,3)
sns_heat = sns.heatmap(scores, annot=True,
            xticklabels=param_grid["nn__hidden_layer_sizes"],
            yticklabels=param_grid["nn__alpha"])
fig = sns_heat.get_figure()
fig.savefig('output/nn_diabetes_importances.pdf')

"""3. Neural Networks Classification"""

#(a) Load the breast cancer dataset using sklearn.datasets.load_breast_cancer()
# As usual, split the data #into test and training set.

cancer = load_breast_cancer()
y = cancer["target"]
X = cancer["data"]
pd.DataFrame(X, columns=cancer['feature_names'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#(c) Learn a a Neural Network Classifier after MinMax-Scaling with in total
# four parameter combinations (and #1000 iterations) of your choice using
# 5-fold Cross-Validation. To keep computation burden low, stop after #1,000
# iterations and use the best solver for this dataset. Using the ROC-AuC-score
# metric to pick the#best model, what are the best parameter combinations,
# which is its ROC-AuC-score, and how well does it #generalize in terms of
# the ROC-AuC-score?

algorithms = [("scaler", MinMaxScaler()),
              ("nn", MLPRegressor(solver="lbfgs", random_state=42,
                                  max_iter=1000))]
pipe = Pipeline(algorithms, verbose=True)
param_grid = {"nn__hidden_layer_sizes": [(75, 75), (100, 100)],
              "nn__alpha": [0.01, 0.001]}
grid = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True,
                    scoring="roc_auc")
grid.fit(X, y)

print(grid.best_params_)
print(grid.best_score_)

# The best parameters are 100,100 and alpha 0.01.
# The ROC-AUC score is .976, which means there is a  97.6% chance
# (discrimination capacity) that the # model will be able to distinguish
# between positive class and negative class.


#(d) Plot the confusion matrix as a heatmap for the best model and save it as
# ./output/nn_breast_confusion.pdf.

mlp = MLPClassifier(max_iter=1000, random_state=42, alpha=0.01,
                    hidden_layer_sizes=(100, 100))
mlp.fit(X_train, y_train)
preds = mlp.predict(X_test)
mlp.score(X_test, y_test)
preds = mlp.predict(X_test)
confusion_m = confusion_matrix(y_test, preds)
sns_heat = sns.heatmap(confusion_m, annot=True)
fig = sns_heat.get_figure()
fig.savefig('./output/nn_breast_confusion.pdf')