function fluxmodel(y::AbstractVector, x::AbstractMatrix; ratio_prediction::Number=0.0, keepcases::BitArray=falses(length(y)), mask_prediction::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, check::Bool=false, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=false, kw...)
end

function xgbmodel(y::AbstractVector, x::AbstractMatrix; ratio_prediction::Number=0.0, keepcases::BitArray=falses(length(y)), mask_prediction::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=false, kw...)
	if isnothing(mask_prediction)
		mask_prediction = SVR.get_prediction_mask(length(y), ratio_prediction; keepcases=keepcases, debug=true)
	else
		@assert length(mask_prediction) == size(x, 1)
		@assert eltype(mask_prediction) <: Bool

		@info("Number of cases for training: $(sum(.!mask_prediction))")
		@info("Number of cases for prediction: $(sum(mask_prediction))")
	end
	if load && isfile(filename)
		@info("Loading XGBoost model from file: $(filename)")
		model = XGBoost.load(filename)
	else
		!quiet && @info("Training ...")
		param_dict = Dict(:max_depth => 20,
			:base_score => 0.5,
			:learning_rate => 0.3,
			:scale_pos_weight => 1,
			:gamma => 0,
			:max_delta_step => 0,
			:subsample => 1.0,
			:colsample_bynode => 1.0,
			:colsample_bytree => 0.9,
			:colsample_bylevel => 0.6,
			:seed => 20,
			:min_child_weight => 1,
			:reg_alpha => 0,y_p
			:reg_lambda => 1,
			:n_estimators => 1000)
		model = XGBoost.xgboost(x[.!mask_prediction, :], 20; label=y[.!mask_prediction], verbose=0, silent=1, param_dict...)
		if save && filename != ""
			@info("Saving model to file: $(filename)")
			Mads.recursivemkdir(filename; filename=true)
			XGBoost.save(filename, model)
		end
	end
	y_pr = XGBoost.predict(model, x)
	@info("Root mean square error (all       ): $(MLJ.rmse(y, y_pr))")
	@info("Root mean square error (training  ): $(MLJ.rmse(y[.!mask_prediction], y_pr[.!mask_prediction]))")
	sum(mask_prediction) > 0 && @info("Root mean square error (prediction): $(MLJ.rmse(y[mask_prediction], y_pr[mask_prediction]))")
	return y_pr, mask_prediction, model
end

function svrmodel(y::AbstractVector, x::AbstractMatrix; ratio_prediction::Number=0.0, keepcases::BitArray=falses(length(y)), mask_prediction::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, epsilon::Float64=0.000000001, gamma::Float64=0.1, check::Bool=false, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=false, kw...)
	if isnothing(mask_prediction)
		mask_prediction = SVR.get_prediction_mask(length(y), ratio_prediction; keepcases=keepcases, debug=true)
	else
		@assert length(mask_prediction) == size(x, 1)
		@assert eltype(mask_prediction) <: Bool
		@info("Number of cases for training: $(sum(.!mask_prediction))")
		@info("Number of cases for prediction: $(sum(mask_prediction))")
	end
	xt = permutedims(x)
	if load && isfile(filename)
		@info("Loading SVR model from file: $(filename)")
		model = SVR.loadmodel(filename)
	else
		!quiet && @info("Training ...")
		model = SVR.train(y[.!mask_prediction], xt[:, .!mask_prediction]; epsilon=epsilon, gamma=gamma)
		if save && filename != ""
			!quiet && @info("Saving model to file: $(filename)")
			Mads.recursivemkdir(filename; filename=true)
			SVR.savemodel(model, filename)
		end
	end
	y_pr = SVR.predict(model, xt)
	if check
		y_pr2, _, _ = SVR.fit_test(y, xt; ratio_prediction=ratio_prediction, quiet=true, mask_prediction=mask_prediction, keepcases=keepcases, epsilon=epsilon, gamma=gamma, kw...)
		@assert vy_pr == vy_pr2
	end
	@info("Root mean square error (all       ): $(MLJ.rmse(y, y_pr))")
	@info("Root mean square error (training  ): $(MLJ.rmse(y[.!mask_prediction], y_pr[.!mask_prediction]))")
	sum(mask_prediction) > 0 && @info("Root mean square error (prediction): $(MLJ.rmse(y[mask_prediction], y_pr[mask_prediction]))")
	return y_pr, mask_prediction, model
end

function mljmodel(y::AbstractVector, x::AbstractMatrix; ratio_prediction::Number=0.0, keepcases::BitArray=falses(length(y)), mask_prediction::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=true, nround::Number=1000, ntries::Number=10, nfolds::Number=5, ml_verbosity::Integer=0, self_tuning::Bool=false,
		ranges::AbstractDict{Symbol,Vector}=OrderedCollections.OrderedDict(:max_depth => [3, 5, 6, 10, 15, 20], :eta => [0.01, 0.1, 0.2, 0.3], :subsample => collect(0.5:0.1:1.0), :colsample_bytree => collect(0.4:0.1:1.0), :colsample_bylevel => collect(0.4:0.1:1.0), :colsample_bynode => collect(0.4:0.1:1.0)),
		model::Any=MLJXGBoostInterface.XGBoostRegressor(; test=1, num_round=nround, booster="gbtree", disable_default_eval_metric=0, eta=0.01, num_parallel_tree=1, gamma=0.0, max_depth=15, min_child_weight=1.0, max_delta_step=0.0, subsample=1.0, colsample_bytree=0.5, colsample_bylevel=0.4, colsample_bynode=1.0, lambda=1.0, alpha=0.0, tree_method="auto", sketch_eps=0.03, scale_pos_weight=1.0, updater=nothing, refresh_leaf=1, process_type="default", grow_policy="depthwise", max_leaves=0, max_bin=256, predictor="cpu_predictor", sample_type="uniform", normalize_type="tree", rate_drop=0.0, one_drop=0, skip_drop=0.0, feature_selector="cyclic", top_k=0, tweedie_variance_power=1.5, objective="reg:squarederror", base_score=0.5, watchlist=nothing, nthread=16, importance_type="gain", seed=nothing, validate_parameters=false, eval_metric=String[]))
	if isnothing(mask_prediction)
		mask_prediction = SVR.get_prediction_mask(length(y), ratio_prediction; keepcases=keepcases, debug=true)
	else
		@assert length(mask_prediction) == size(x, 1)
		@assert eltype(mask_prediction) <: Bool
		@info("Number of cases for training: $(sum(.!mask_prediction))")
		@info("Number of cases for prediction: $(sum(mask_prediction))")
	end
	if load && isfile(filename)
		@info("Loading MLJ model from file: $(filename)")
		mlj_machine = MLJ.machine(filename)
	else
		if self_tuning
			@info("Self-Tuning & Training ...")
			ranges_vector = []
			for k in keys(ranges)
				push!(ranges_vector, MLJ.range(model, k; values=ranges[k]))
				@info("Parameter $(k): range = $(ranges[k])")
			end
			self_tuning_model = MLJ.TunedModel(; model=model, n=ntries, resampling=MLJ.CV(; nfolds=nfolds), tuning=MLJ.RandomSearch(), range=ranges_vector, measure=MLJ.rmse, compact_history=true)
			mlj_machine = MLJ.machine(self_tuning_model, MLJ.table(x[.!mask_prediction,:]), y[.!mask_prediction])
			MLJ.fit!(mlj_machine; verbosity=ml_verbosity)
			for k in keys(ranges)
				@info("Parameter $(k): range = $(ranges[k]) => estimate = $(getproperty(MLJ.fitted_params(mlj_machine).best_model, k))")
			end
		else
			!quiet && @info("Training ...")
			mlj_machine = MLJ.machine(model, MLJ.table(x[.!mask_prediction,:]), y[.!mask_prediction])
			MLJ.fit!(mlj_machine; verbosity=ml_verbosity)
		end
		if save && filename != ""
			@info("Saving MLJ model to file: $(filename)")
			Mads.recursivemkdir(filename; filename=true)
			MLJ.save(filename, mlj_machine)
		end
	end
	y_pr = MLJ.predict(mlj_machine, MLJ.table(x))
	@info("Root mean square error (all       ): $(MLJ.rmse(y, y_pr))")
	@info("Root mean square error (training  ): $(MLJ.rmse(y[.!mask_prediction], y_pr[.!mask_prediction]))")
	sum(mask_prediction) > 0 && @info("Root mean square error (prediction): $(MLJ.rmse(y[mask_prediction], y_pr[mask_prediction]))")
	return y_pr, mask_prediction, mlj_machine
end

function predict(mlj_machine::MLJ.Machine, x::AbstractArray)
	y_pr = MLJ.predict(mlj_machine, MLJ.table(x))
	return y_pr
end

function predict(xgb_machine::XGBoost.Booster, x::AbstractArray)
	y_pr = XGBoost.predict(xgb_machine, x)
	return y_pr
end

function predict(svr_machine::SVR.svmmodel, x::AbstractArray)
	y_pr = SVR.predict(svr_machine, permutedims(x))
	return y_pr
end

function save(filename::AbstractString, mlj_machine::MLJ.Machine)
	Mads.recursivemkdir(filename; filename=true)
	MLJ.save(filename, mlj_machine)
	return nothing
end

function save(filename::AbstractString, xgb_machine::XGBoost.Booster)
	Mads.recursivemkdir(filename; filename=true)
	XGBoost.save(filename, xgb_machine)
	return nothing
end

function save(filename::AbstractString, svr_machine::SVR.svmmodel)
	Mads.recursivemkdir(filename; filename=true)
	SVR.save(svr_machine, filename)
	return nothing
end