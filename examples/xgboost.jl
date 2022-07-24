import CSV
import DataFrames
import ScikitLearn
@ScikitLearn.sk_import ensemble: RandomForestRegressor
import XGBoost
import PyCall
xgb = PyCall.pyimport("xgboost")
np = PyCall.pyimport("numpy")

cd(@__DIR__)

x = CSV.read("x.csv", DataFrames.DataFrame)
y = CSV.read("y.csv", DataFrames.DataFrame)

mod = RandomForestRegressor(max_leaf_nodes=2)
param_dict = Dict("n_estimators"=>[50, 100, 200, 300], "max_depth"=> [3, 5, 6, 8, 9, 10])
model = ScikitLearn.GridSearch.RandomizedSearchCV(mod, param_dist; n_iter=10, cv=5, n_jobs=1)
ScikitLearn.fit!(model, Matrix(x), Matrix(y))
xgb_model = model.best_estimator_
xgb_model.fit(Matrix(x), Matrix(y))
y_pr = xgb_model.predict(Matrix(x))

mod = xgb.XGBRegressor(seed=20)
param_dict = Dict("max_depth"=>[3, 5, 6, 10, 15, 20],
	"learning_rate"=>[0.01, 0.1, 0.2, 0.3],
	"subsample"=>collect(0.5:0.1:1.0),
	"colsample_bytree"=>collect(0.4:0.1:1.0),
	"colsample_bylevel"=>collect(0.4:0.1:1.0),
	"n_estimators"=>[100, 500, 1000])
model = ScikitLearn.GridSearch.RandomizedSearchCV(mod, param_dict; verbose=1, n_jobs=1, n_iter=10, cv=5)
ScikitLearn.fit!(model, Matrix(x), Matrix(y)[:,1])
xgb_model = model.best_estimator_
xgb_model.fit(Matrix(x), Matrix(y)[:,1])
y_pr = xgb_model.predict(Matrix(x))
NMFk.plotscatter(y_pr, Matrix(y)[:,1])

PyCall.py"""
import numpy as np
import xgboost
from sklearn.model_selection import RandomizedSearchCV

def run_xgb_model(X_train, X_val, y, y_val):
	params = {	'max_depth': [3, 5, 6, 10, 15, 20],
				'learning_rate': [0.01, 0.1, 0.2, 0.3],
				'subsample': np.arange(0.5, 1.0, 0.1),
				'colsample_bytree': np.arange(0.4, 1.0, 0.1),
				'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
				'n_estimators': [100, 500, 1000]}
	
	xgbr = xgboost.XGBRegressor(seed=20)
	
	clf = RandomizedSearchCV(estimator=xgbr,
							param_distributions=params,
							scoring='r2',
							n_iter=25,
							verbose=1,
							return_train_score=True)
	clf.fit(X_train, y)
	
	xgb_model = clf.best_estimator_
	xgb_model.fit(X_train, y)
	y_pred = xgb_model.predict(X_val)
	scores = [clf.cv_results_["mean_train_score"].mean(), clf.cv_results_['std_train_score'].mean()]
	print("Best parameters:", clf.best_params_)
	print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
	return y_pred, xgb_model
"""

PyCall.py"run_xgb_model"(Matrix(x), Matrix(x), Matrix(y)[:,1], Matrix(y)[:,1])