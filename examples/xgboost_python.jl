function xgbpymodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0.0, keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, load::Bool=false, save::Bool=false, filemodel::AbstractString="", quiet::Bool=false, kw...)
	if pm === nothing
		pm = SVR.get_prediction_mask(length(y), ratio; keepcases=keepcases, debug=true)
	else
		@assert length(pm) == size(x, 1)
		@assert eltype(pm) <: Bool
	end
	if load && isfile(filemodel)
		@info("Loading XGBoost model from file: $(filemodel)")
		xgb_model = XGBoost.load(filemodel)
	else
		!quiet && @info("Training ...")
		xgb = PyCall.pyimport("xgboost")
		mod = xgb.XGBRegressor(seed=20)
		param_dict = Dict("max_depth" => [3, 5, 6, 10, 15, 20],
			"learning_rate" => [0.01, 0.1, 0.2, 0.3],
			"subsample" => collect(0.5:0.1:1.0),
			"colsample_bytree" => collect(0.4:0.1:1.0),
			"colsample_bylevel" => collect(0.4:0.1:1.0),
			"n_estimators" => [100, 500, 1000])
		model = ScikitLearn.GridSearch.RandomizedSearchCV(mod, param_dict; verbose=1, n_jobs=1, n_iter=10, cv=5)
		ScikitLearn.fit!(model, x[.!pm, :], y[.!pm])
		xgb_model = model.best_estimator_
		xgb_model.fit(x[.!pm, :], y[.!pm])
		if save && filemodel != ""
			@info("Saving model to file: $(filemodel)")
			Mads.recursivemkdir(filemodel; filename=true)
			XGBoost.save(filemodel, xgb_model)
		end
	end
	y_pr = xgb_model.predict(x)
	return y_pr, pm, xgb_model
end