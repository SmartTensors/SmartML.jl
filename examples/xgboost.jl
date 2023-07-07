import MLJ
import MLJXGBoostInterface
import MLJLIBSVMInterface
import XGBoost
import CSV
import DataFrames
import NMFk
import Mads
import ScikitLearn
@ScikitLearn.sk_import ensemble: RandomForestRegressor
import PyCall
xgb = PyCall.pyimport("xgboost")
np = PyCall.pyimport("numpy")

cd(@__DIR__)

x = CSV.read("x.csv", DataFrames.DataFrame)
y = CSV.read("y.csv", DataFrames.DataFrame)

# Python ScikitLearn XGBoost
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
	y_pr = xgb_model.predict(X_val)
	scores = [clf.cv_results_["mean_train_score"].mean(), clf.cv_results_['std_train_score'].mean()]
	print("Best parameters:", clf.best_params_)
	print("Lowest RMSE: ", (-clf.best_score_)**(1/2.0))
	return y_pr, xgb_model
"""
y_pr, pyskl_model_xgb_best = PyCall.py"run_xgb_model"(Matrix(x), Matrix(x), Matrix(y)[:,1], Matrix(y)[:,1])
NMFk.plotscatter(y_pr, Matrix(y)[:,1])

# Julia ScikitLearn
# RF
skl_model_rf = RandomForestRegressor(; max_leaf_nodes=2)
param_dict = Dict("n_estimators"=>[50, 100, 200, 300], "max_depth"=> [3, 5, 6, 8, 9, 10])
skl_model_rf_self_training = ScikitLearn.GridSearch.RandomizedSearchCV(skl_model_rf, param_dict; n_iter=10, cv=5, n_jobs=1)
ScikitLearn.fit!(skl_model_rf_self_training, Matrix(x), Matrix(y))
skl_model_rf_best = skl_model_rf_self_training.best_estimator_
skl_model_rf_best.fit(Matrix(x), Matrix(y))
y_pr = skl_model_rf_best.predict(Matrix(x))
NMFk.plotscatter(y_pr[:,1], Matrix(y)[:,1])

# Julia ScikitLearn
# XGBoost
skl_model_xgb = xgb.XGBRegressor(; seed=20)
param_dict = Dict("max_depth"=>[3, 5, 6, 10, 15, 20],
	"learning_rate"=>[0.01, 0.1, 0.2, 0.3],
	"subsample"=>collect(0.5:0.1:1.0),
	"colsample_bytree"=>collect(0.4:0.1:1.0),
	"colsample_bylevel"=>collect(0.4:0.1:1.0),
	"n_estimators"=>[100, 500, 1000])
skl_model_xgb_self_training = ScikitLearn.GridSearch.RandomizedSearchCV(skl_model_xgb, param_dict; verbose=1, n_jobs=1, n_iter=25, cv=5)
ScikitLearn.fit!(skl_model_xgb_self_training, Matrix(x), Matrix(y)[:,1])
skl_model_xgb_best = skl_model_xgb_self_training.best_estimator_
skl_model_xgb_best.fit(Matrix(x), Matrix(y)[:,1])
y_pr = skl_model_xgb_best.predict(Matrix(x))
NMFk.plotscatter(y_pr, Matrix(y)[:,1])

# Julia XGBoost
model = XGBoost.xgboost((Matrix(x), Matrix(y)[:,1]); XGBoost.regression(test = 1,
			num_round = 100,
			booster = "gbtree",
			disable_default_eval_metric = 0,
			eta = 0.3,
			num_parallel_tree = 1,
			gamma = 0.0,
			max_depth = 6,
			min_child_weight = 1.0,
			max_delta_step = 0.0,
			subsample = 1.0,
			colsample_bytree = 1.0,
			colsample_bylevel = 1.0,
			colsample_bynode = 1.0,
			lambda = 1.0,
			alpha = 0.0,
			tree_method = "auto",
			sketch_eps = 0.03,
			scale_pos_weight = 1.0,
			refresh_leaf = 1,
			process_type = "default",
			grow_policy = "depthwise",
			max_leaves = 0,
			max_bin = 256,
			predictor = "cpu_predictor",
			sample_type = "uniform",
			normalize_type = "tree",
			rate_drop = 0.0,
			one_drop = 0,
			skip_drop = 0.0,
			feature_selector = "cyclic",
			top_k = 0,
			tweedie_variance_power = 1.5,
			objective = "reg:squarederror",
			base_score = 0.5,
			importance_type = "gain",
			validate_parameters = false)...)
y_pr = XGBoost.predict(model, Matrix(x))
NMFk.plotscatter(y_pr, Matrix(y)[:,1])

# MLJ XGBoost
XGBModel = MLJ.@load XGBoostRegressor verbosity=1
xgbmodel = XGBModel()

r_max_depth = MLJ.range(xgbmodel, :max_depth; values=[3, 5, 6, 10, 15, 20])
r_eta = MLJ.range(xgbmodel, :eta; values=[0.01, 0.1, 0.2, 0.3])
r_subsample = MLJ.range(xgbmodel, :subsample; values=collect(0.5:0.1:1.0))
r_colsample_bytree = MLJ.range(xgbmodel, :colsample_bytree; values=collect(0.4:0.1:1.0))
r_colsample_bylevel = MLJ.range(xgbmodel, :colsample_bylevel; values=collect(0.4:0.1:1.0))
r_num_boost_round = MLJ.range(xgbmodel, :num_boost_round; values=[100, 500, 1000])

self_tuning_xgbmodel = MLJ.TunedModel(model=xgbmodel, resampling=MLJ.CV(nfolds=5), tuning=MLJ.RandomSearch(), range=[r_max_depth, r_eta, r_subsample, r_colsample_bytree, r_colsample_bylevel, r_num_boost_round], measure=MLJ.rms)

xgb_machine = MLJ.machine(self_tuning_xgbmodel, MLJ.table(Matrix(x)), Matrix(y)[:,1])
MLJ.fit!(xgb_machine; verbosity=1)

y_pr = MLJ.predict(mach, MLJ.table(Matrix(x)))
NMFk.plotscatter(y_pr, Matrix(y)[:,1])

MLJ.fitted_params(mach).best_model

mach = MLJ.machine(self_tuning_xgbmodel, MLJ.table(GIMI.X), GIMI.y)
MLJ.fit!(mach; verbosity=1)

y_pr = MLJ.predict(mach, MLJ.table(Matrix(x)))
NMFk.plotscatter(y_pr, Matrix(y)[:,1])

# MLJ SVR
SVRModel = MLJ.@load EpsilonSVR verbosity=1
svrmodel = SVRModel()
