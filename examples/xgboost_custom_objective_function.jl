import XGBoost

const DATAPATH = joinpath(first(splitdir(first(splitdir(pathof(XGBoost))))), "data")
dtrain = XGBoost.DMatrix(joinpath(DATAPATH, "agaricus.txt.train"))
dtest = XGBoost.DMatrix(joinpath(DATAPATH, "agaricus.txt.test"))

# NOTE: for a customized objective function, we leave objective as a default
# NOTE: what we are getting is margin value in prediction
function logregobj(preds::Vector{Float32}, dtrain::DMatrix)
	labels = get_info(dtrain, "label")
	preds = 1.0 ./ (1.0 .+ exp.(-preds))
	grad = preds .- labels
	hess = preds .* (1.0 .- preds)
	return (grad, hess)
end

# user defined evaluation function, return a pair metric_name, result
# NOTE: when you use a customized loss function, the default prediction value is margin
# this may make the build-in evaluation metric not function properly
# for example, we are doing logistic loss
# the prediction is the score before logistic transformation
# the build-in evaluation error assumes the input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need to write a customized evaluation function
function evalerror(preds::Vector{Float32}, dtrain::DMatrix)
	labels = get_info(dtrain, "label")
	# return a pair metric_name, result
	# since preds are margin (before logistic transformation, cutoff at 0)
	return ("self-error", sum((preds .> 0.0) .!= labels) / float(size(preds, 1)))
end

param = ["max_depth"=>2, "eta"=>3, "silent"=>1, "verbose"=>10]
watchlist  = [(dtest, "eval"), (dtrain, "train")]
num_round = 10

# training with a customized objective function
# we can also do step-by-step training
# check the xgboost_lib.jl's implementation of train
bst = XGBoost.xgboost(dtrain, num_round; param=param, watchlist=watchlist, obj=logregobj, feval=evalerror)