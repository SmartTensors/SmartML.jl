import MLJ
import MLJXGBoostInterface
import MLJLIBSVMInterface
import XGBoost
import CSV
import JLD
import DataFrames
import NMFk
import Mads
import ScikitLearn
@ScikitLearn.sk_import ensemble: RandomForestRegressor
import PyCall
xgb = PyCall.pyimport("xgboost")
np = PyCall.pyimport("numpy")

cd(@__DIR__)

x = JLD.load("gimi-x.jld", "x")
y = JLD.load("gimi-y.jld", "y")


# Julia XGBoost
xgb_model = XGBoost.xgboost((x, y); XGBoost.regression(test = 1,
  num_round = 1000,
  booster = "gbtree",
  disable_default_eval_metric = 0,
  eta = 0.01,
  num_parallel_tree = 1,
  gamma = 0.0,
  max_depth = 15,
  min_child_weight = 1.0,
  max_delta_step = 0.0,
  subsample = 1.0,
  colsample_bytree = 0.5,
  colsample_bylevel = 0.4,
  colsample_bynode = 1.0,
  lambda = 1.0,
  alpha = 0.0,
  tree_method = "auto",
  sketch_eps = 0.03,
  scale_pos_weight = 1.0,
  updater = nothing,
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
  watchlist = nothing,
  nthread = 16,
  importance_type = "gain",
  seed = nothing,
  validate_parameters = false,
  eval_metric = String[])...)
y_pr = XGBoost.predict(xgb_model, x)
NMFk.plotscatter(y_pr, y)

# MLJ XGBoost
XGBModel = MLJ.@load XGBoostRegressor verbosity=1
xgbmodel = XGBModel()

r_max_depth = MLJ.range(xgbmodel, :max_depth; values=[3, 5, 6, 10, 15, 20])
r_eta = MLJ.range(xgbmodel, :eta; values=[0.01, 0.1, 0.2, 0.3])
r_subsample = MLJ.range(xgbmodel, :subsample; values=collect(0.5:0.1:1.0))
r_colsample_bytree = MLJ.range(xgbmodel, :colsample_bytree; values=collect(0.4:0.1:1.0))
r_colsample_bylevel = MLJ.range(xgbmodel, :colsample_bylevel; values=collect(0.4:0.1:1.0))
r_num_round = MLJ.range(xgbmodel, :num_round; values=[100, 500, 1000])

self_tuning_xgbmodel = MLJ.TunedModel(model=xgbmodel, resampling=MLJ.CV(nfolds=5), tuning=MLJ.RandomSearch(), range=[r_max_depth, r_eta, r_subsample, r_colsample_bytree, r_colsample_bylevel, r_num_round], measure=MLJ.rms)

xgb_machine = MLJ.machine(self_tuning_xgbmodel, MLJ.table(x), y)
MLJ.fit!(xgb_machine; verbosity=1)

y_pr = MLJ.predict(xgb_machine, MLJ.table(x))
NMFk.plotscatter(y_pr, y)

mlj_params = MLJ.fitted_params(xgb_machine).best_model
xgbmodel_fit = XGBModel(test = 1,
num_round = 1000,
booster = "gbtree",
disable_default_eval_metric = 0,
eta = 0.01,
num_parallel_tree = 1,
gamma = 0.0,
max_depth = 15,
min_child_weight = 1.0,
max_delta_step = 0.0,
subsample = 1.0,
colsample_bytree = 0.5,
colsample_bylevel = 0.4,
colsample_bynode = 1.0,
lambda = 1.0,
alpha = 0.0,
tree_method = "auto",
sketch_eps = 0.03,
scale_pos_weight = 1.0,
updater = nothing,
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
watchlist = nothing,
nthread = 16,
importance_type = "gain",
seed = nothing,
validate_parameters = false,
eval_metric = String[])
xgb_machine_fit = MLJ.machine(xgbmodel_fit, MLJ.table(x), y)
MLJ.fit!(xgb_machine_fit; verbosity=1)
y_pr = MLJ.predict(xgb_machine_fit, MLJ.table(x))
NMFk.plotscatter(y_pr, y)

# MLJ SVR
SVRModel = MLJ.@load EpsilonSVR verbosity=1
svrmodel = SVRModel()
r_epsilon = MLJ.range(svrmodel, :epsilon; lower=1e-12, upper=1e12, scale=:log10)
r_gamma = MLJ.range(svrmodel, :gamma; lower=1e-6, upper=1e6, scale=:log10)
self_tuning_svrmodel = MLJ.TunedModel(model=svrmodel, resampling=MLJ.CV(nfolds=50), tuning=MLJ.Grid(), range=[r_epsilon, r_gamma], measure=MLJ.rms)
svr_machine = MLJ.machine(self_tuning_svrmodel, MLJ.table(x), y)
MLJ.fit!(svr_machine; verbosity=1)
y_pr = MLJ.predict(svr_machine, MLJ.table(x))
NMFk.plotscatter(y_pr, y)
MLJ.fitted_params(svr_machine).best_model

svrmodel_n = SVRModel(gamma = 0.01, epsilon = 1e-12)
svr_machine_n = MLJ.machine(svrmodel_n, MLJ.table(x), y)
MLJ.fit!(svr_machine_n; verbosity=1)
y_pr = MLJ.predict(svr_machine_n, MLJ.table(x))
NMFk.plotscatter(y_pr, y)

xgbmodel_fit = XGBModel(test = 1, num_round = 1000, booster = "gbtree", disable_default_eval_metric = 0, eta = 0.5, num_parallel_tree = 1, gamma = 0.0, max_depth = 3, min_child_weight = 1.0, max_delta_step = 0.0, subsample = 1.0, colsample_bytree = 0.9, colsample_bylevel = 0.4, colsample_bynode = 1.0, lambda = 1.0, alpha = 0.0, tree_method = "auto", sketch_eps = 0.03, scale_pos_weight = 1.0, updater = nothing, refresh_leaf = 1, process_type = "default", grow_policy = "depthwise", max_leaves = 0, max_bin = 256, predictor = "cpu_predictor", sample_type = "uniform", normalize_type = "tree", rate_drop = 0.0, one_drop = 0, skip_drop = 0.0, feature_selector = "cyclic", top_k = 0, tweedie_variance_power = 1.5, objective = "reg:squarederror", base_score = 0.5, watchlist = nothing, nthread = 16, importance_type = "gain", seed = nothing, validate_parameters = false, eval_metric = String[])
xgb_machine_fit = MLJ.machine(xgbmodel_fit, MLJ.table(x), y)
MLJ.fit!(xgb_machine_fit; verbosity=1)
y_pr = MLJ.predict(xgb_machine_fit, MLJ.table(x))
NMFk.plotscatter(y_pr, y)

svrmodel_n = SVRModel(; gamma=0.1, epsilon=0.000000001)
svr_machine_n = MLJ.machine(svrmodel_n, MLJ.table(x), y)
MLJ.fit!(svr_machine_n; verbosity=1)
y_pr = MLJ.predict(svr_machine_n, MLJ.table(x))
NMFk.plotscatter(y_pr, y)

