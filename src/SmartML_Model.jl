import MLJ
import MLJXGBoostInterface
import MLJLIBSVMInterface
import XGBoost
import Mads
import NMFk
import SVR
import Printf
import Suppressor

smlmodel = Union{SVR.svmmodel, MLJ.Machine}

function setdata(Xin::AbstractMatrix, Xt::AbstractMatrix; order=Colon(), mask=Colon(), quiet::Bool=false)
	ntimes = size(Xt, 1)
	ncases = size(Xin, 1)
	T = [repeat(Xt; inner=(ncases, 1)) repeat(Xin[order, mask], ntimes)]
	if !quiet
		@info("Number of training cases: $(ncases)")
		@info("Number of training times: $(ntimes)")
		@info("Number of training cases * times: $(size(T, 1))")
		@info("Number of training parameters: $(size(T, 2))")
	end
	return T
end

function setdata(Xin::AbstractMatrix, times::AbstractVector; order=Colon(), mask=Colon(), quiet::Bool=false)
	ntimes = length(times)
	ncases = size(Xin, 1)
	T = [repeat(times; inner=ncases) repeat(Xin[order, mask], ntimes)]
	if !quiet
		@info("Number of training cases: $(ncases)")
		@info("Number of training times: $(ntimes)")
		@info("Number of training cases * times: $(size(T, 1))")
		@info("Number of training parameters: $(size(T, 2))")
	end
	return T
end

function setup_mask(ratio::Number, keepcases::BitArray, ncases, ntimes, ptimes::Union{Vector{Integer},AbstractUnitRange})
	pm = SVR.get_prediction_mask(ncases, ratio; keepcases=keepcases)
	lpm = Vector{Bool}(undef, 0)
	for i = 1:ntimes
		opm = (i in ptimes) ? pm : falses(length(pm))
		lpm = vcat(lpm, opm)
	end
	return pm, lpm
end

function fluxmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0.0, keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, check::Bool=false, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=false, kw...)
end

function xgbmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0.0, keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=false, kw...)
	if isnothing(pm)
		pm = SVR.get_prediction_mask(length(y), ratio; keepcases=keepcases, debug=true)
	else
		@assert length(pm) == size(x, 1)
		@assert eltype(pm) <: Bool
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
			:reg_alpha => 0,
			:reg_lambda => 1,
			:n_estimators => 1000)
		model = XGBoost.xgboost(x[.!pm, :], 20; label=y[.!pm], verbose=0, silent=1, param_dict...)
		if save && filename != ""
			@info("Saving model to file: $(filename)")
			Mads.recursivemkdir(filename; filename=true)
			XGBoost.save(filename, model)
		end
	end
	y_pr = XGBoost.predict(model, x)
	return y_pr, pm, model
end

function svrmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0.0, keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, epsilon::Float64=0.000000001, gamma::Float64=0.1, check::Bool=false, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=false, kw...)
	if isnothing(pm)
		pm = SVR.get_prediction_mask(length(y), ratio; keepcases=keepcases, debug=true)
	else
		@assert length(pm) == size(x, 1)
		@assert eltype(pm) <: Bool
	end
	xt = permutedims(x)
	if load && isfile(filename)
		@info("Loading SVR model from file: $(filename)")
		model = SVR.loadmodel(filename)
	else
		!quiet && @info("Training ...")
		model = SVR.train(y[.!pm], xt[:, .!pm]; epsilon=epsilon, gamma=gamma)
		if save && filename != ""
			!quiet && @info("Saving model to file: $(filename)")
			Mads.recursivemkdir(filename; filename=true)
			SVR.savemodel(model, filename)
		end
	end
	y_pr = SVR.predict(model, xt)
	if check
		y_pr2, _, _ = SVR.fit_test(y, xt; ratio=ratio, quiet=true, pm=pm, keepcases=keepcases, epsilon=epsilon, gamma=gamma, kw...)
		@assert vy_pr == vy_pr2
	end
	return y_pr, pm, model
end

function mljmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0.0, keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, load::Bool=false, save::Bool=false, filename::AbstractString="", quiet::Bool=true, MLJmodel::DataType=MLJXGBoostInterface.XGBoostRegressor, ml_verbosity::Integer=0, self_tuning::Bool=false, kw...)
	x_table = MLJ.table(x)
	if isnothing(pm)
		pm = SVR.get_prediction_mask(length(y), ratio; keepcases=keepcases, debug=true)
	else
		@assert length(pm) == size(x, 1)
		@assert eltype(pm) <: Bool
	end
	if load && isfile(filename)
		@info("Loading MLJ model from file: $(filename)")
		mlj_machine = MLJ.machine(filename)
	else
		if self_tuning
			!quiet && @info("Self-Tuning & Training ...")
			mljmodel = MLJmodel()
			if ml_method == "XGBoostRegressor"
				r_max_depth = MLJ.range(mljmodel, :max_depth; values=[3, 5, 6, 10, 15, 20])
				r_eta = MLJ.range(mljmodel, :eta; values=[0.01, 0.1, 0.2, 0.3])
				r_subsample = MLJ.range(mljmodel, :subsample; values=collect(0.5:0.1:1.0))
				r_colsample_bytree = MLJ.range(mljmodel, :colsample_bytree; values=collect(0.4:0.1:1.0))
				r_colsample_bylevel = MLJ.range(mljmodel, :colsample_bylevel; values=collect(0.4:0.1:1.0))
				r_num_round = MLJ.range(mljmodel, :num_round; values=[100, 500, 1000])
				range_v = [r_max_depth, r_eta, r_subsample, r_colsample_bytree, r_colsample_bylevel, r_num_round]
			end
			self_tuning_mljmodel = MLJ.TunedModel(model=xgbmodel, resampling=MLJ.CV(nfolds=5), tuning=MLJ.RandomSearch(), range=range_v, measure=MLJ.rms)
			mlj_machine = MLJ.machine(self_tuning_mljmodel, x_table, y)
			MLJ.fit!(mlj_machine; verbosity=ml_verbosity)
		else
			!quiet && @info("Training ...")
			mljmodel = MLJmodel(test=1, num_round=1000, booster="gbtree", disable_default_eval_metric=0, eta=0.01, num_parallel_tree=1, gamma=0.0, max_depth=15, min_child_weight=1.0, max_delta_step=0.0, subsample=1.0, colsample_bytree=0.5, colsample_bylevel=0.4, colsample_bynode=1.0, lambda=1.0, alpha=0.0, tree_method="auto", sketch_eps=0.03, scale_pos_weight=1.0, updater=nothing, refresh_leaf=1, process_type="default", grow_policy="depthwise", max_leaves=0, max_bin=256, predictor="cpu_predictor", sample_type="uniform", normalize_type="tree", rate_drop=0.0, one_drop=0, skip_drop=0.0, feature_selector="cyclic", top_k=0, tweedie_variance_power=1.5, objective="reg:squarederror", base_score=0.5, watchlist=nothing, nthread=16, importance_type="gain", seed=nothing, validate_parameters=false, eval_metric=String[])
			mlj_machine = MLJ.machine(mljmodel, x_table, y)
			MLJ.fit!(mlj_machine; verbosity=ml_verbosity)
		end
		if save && filename != ""
			@info("Saving MLJ model to file: $(filename)")
			Mads.recursivemkdir(filename; filename=true)
			MLJ.save(filename, xgb_machine_fit)
		end
	end
	y_pr = MLJ.predict(mlj_machine, x_table)
	return y_pr, pm, mlj_machine
end

function predict(mlj_machine::MLJ.Machine, x::AbstractMatrix)
	y_pr = MLJ.predict(mlj_machine, MLJ.table(x))
	return y_pr
end

function predict(xgb_machine::XGBoost.Booster, x::AbstractMatrix)
	y_pr = XGBoost.predict(xgb_machine, x)
	return y_pr
end

function predict(svr_machine::SVR.svmmodel, x::AbstractMatrix)
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

function model(Xo::AbstractMatrix, Xi::AbstractMatrix, times::AbstractVector=Vector(undef, 0), Xtn::AbstractMatrix=Matrix(undef, 0, 0); keepcases::BitArray=falses(size(Xo, 1)), modeltype::Symbol=:svr, ratio::Number=0, ptimes::Union{Vector{Integer},AbstractUnitRange}=1:length(times), plot::Bool=false, plottime::Bool=false, mask=Colon(), load::Bool=false, save::Bool=false, modeldir::AbstractString=joinpath(workdir, "model_$(modeltype)"), plotdir::AbstractString=joinpath(workdir, "figures_$(modeltype)"), case::AbstractString="", filename::AbstractString="", quiet::Bool=false, kw...)
	inan = vec(.!isnan.(sum(Xo; dims=2))) .|| vec(.!isnan.(sum(Xi; dims=2)))
	Xon, Xomin, Xomax, Xob = NMFk.normalizematrix_col(Xo[inan, :])
	Xin, Ximin, Ximax, Xib = NMFk.normalizematrix_col(Xi[inan, :])
	ntimes = length(times)
	ncases = size(Xin, 1)
	if sizeof(Xtn) > 0
		@assert size(Xtn, 1) == ntimes
		tn, tmin, tmax = NMFk.normalize(Float64.(times))
		T = setdata(Xin, [tn Xtn]; mask=mask, quiet=quiet)
	elseif ntimes > 0
		tn, tmin, tmax = NMFk.normalize(Float64.(times))
		T = setdata(Xin, tn; mask=mask, quiet=quiet)
	else
		@assert size(Xon, 2) == 1
		T = Xin[mask, :]
	end
	if ntimes > 0
		vy_trn = vec(Xon[:, 1:ntimes])
		pm, lpm = setup_mask(ratio, keepcases, ncases, ntimes, ptimes)
	else
		vy_trn = vec(Xon)
		pm = SVR.get_prediction_mask(length(vy_trn), ratio; keepcases=keepcases, debug=false)
		lpm = pm
	end
	if !quiet
		@info("Number of cases for training: $(ncases - sum(pm))")
		@info("Number of cases for prediction: $(sum(pm))")
		if ntimes > 0
			@info("Number of cases/transients for training: $(ncases * ntimes - sum(lpm))")
			@info("Number of cases/transients for prediction: $(sum(lpm))")
		end
	end
	if plot && ntimes > 0
		Mads.plotseries(Xo[.!pm, 1:ntimes]', "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_training_series.png"; title="Training set ($(sum(.!pm)))", xaxis=times, xmin=0, xmax=times[end])
		if sum(pm) > 0
			Mads.plotseries(Xo[pm, 1:ntimes]', "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_prediction_series.png"; title="Prediction set ($(sum(pm)))", xaxis=times, xmin=0, xmax=times[end])
		end
	end
	if (load || save) && (filename != "" || case != "")
		filename = joinpath(modeldir, "$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm)).$(modeltype)model")
	end
	if modeltype == :svr
		vy_prn, _, model = svrmodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet, kw...)
	elseif modeltype == :xgb
		vy_prn, _, model = xgbmodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet)
	elseif modeltype == :xgbpy
		vy_prn, _, model = xgbpymodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet)
	elseif modeltype == :flux
		vy_prn, _, model = fluxmodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet)
	elseif modeltype == :piml
		vy_prn, _, model = pimlmodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet)
	else
		@warn("Unknown model type! SVR will be used!")
		vy_prn, _, model = svrmodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet, kw...)
	end
	if ntimes > 0
		vy_prn = reshape(vy_prn, ncases, ntimes)
		vy_trn = reshape(vy_trn, ncases, ntimes)
	else
		vy_prn = reshape(vy_prn, ncases, 1)
		vy_trn = reshape(vy_trn, ncases, 1)
	end
	vy_prn[vy_prn.<0] .= 0
	vy_pr = vec(NMFk.denormalize(vy_prn, Xomin, Xomax))
	vy_tr = vec(NMFk.denormalize(vy_trn, Xomin, Xomax))
	aimin = vec(Ximin)
	aimax = vec(Ximax)
	aomin = vec(Xomin)
	aomax = vec(Xomax)
	function mlmodel(x::AbstractVector)
		xn = first(NMFk.normalize(x; amin=aimin, amax=aimax))
		if ntimes > 0
			y = Vector{Float64}(undef, ntimes)
			for t = 1:ntimes
				y[t] = first(SVR.predict(model, [tn[t]; xn]))
			end
		else
			y = SVR.predict(model, xn)
		end
		NMFk.denormalize!(y, aomin, aomax)
		return y
	end
	function mlmodel(X::AbstractMatrix)
		nc = size(X, 1)
		Xn, _, _, _ = NMFk.normalizematrix_col(X; amin=Ximin, amax=Ximax)
		if ntimes > 0
			Yn = Matrix{Float64}(undef, nc, ntimes)
			for t = 1:ntimes
				S = [repeat(tn[t:t], nc) Xn]
				Yn[:, t] = SVR.predict(model, S')
			end
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
		else
			Yn = SVR.predict(model, Xn')
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
			Y = reshape(Y, nc, 1)
		end
		return Y
	end
	r2t = SVR.r2(vy_tr[.!lpm], vy_pr[.!lpm])
	rmset = NMFk.rmsenan(vy_tr[.!lpm], vy_pr[.!lpm])
	!quiet && println("Training $(ncases - sum(pm)) / $(sum(pm)) $(ratio) r2 = $(round(r2t; sigdigits=3)) rmse = $(round(rmset; sigdigits=3))")
	if sum(lpm) > 0
		r2p = SVR.r2(vy_tr[lpm], vy_pr[lpm])
		rmsep = NMFk.rmsenan(vy_tr[lpm], vy_pr[lpm])
		!quiet && println("Prediction $(ncases - sum(pm)) / $(sum(pm)) $(ratio) r2 = $(round(r2p; sigdigits=3)) rmse = $(round(rmsep; sigdigits=3))")
	else
		r2p = rmsep = NaN
	end
	if quiet
		println("$(ncases - sum(pm)) / $(sum(pm)) $(ratio) $(r2t) $(r2p) $(rmset) $(rmsep)")
	end
	if plot
		Mads.plotseries([vy_tr vy_pr], "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_series.png"; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
		NMFk.plotscatter(vy_tr[.!lpm], vy_pr[.!lpm]; title="Training Size: $(sum(.!pm)); r<sup>2</sup>: $(round(r2t; sigdigits=3))", xtitle="True", ytitle="Estimate", filename="$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_scatter_training.png")

	end
	if plot && sum(lpm) > 0
		NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Prediction Size: $(sum(pm)); r<sup>2</sup>: $(round(r2p; sigdigits=3))", xtitle="True", ytitle="Estimate", filename="$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_scatter_prediction.png")
	end
	if ntimes > 0
		y_pr = reshape(vy_pr, ncases, ntimes)
	else
		y_pr = reshape(vy_pr, ncases, 1)
	end
	if plottime && ntimes > 0
		for i = 1:ncases
			Mads.plotseries(permutedims([Xo[i:ncases:end, :]; y_pr[i:ncases:end, :]]), "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_transient.png"; xaxis=times, xmin=0, xmax=times[end], names=["Truth", "Prediction"])
		end
	end
	return mlmodel, model, y_pr, T
end

function sensitivity(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector, keepcases::BitArray, attributes::AbstractVector; kw...)
	@assert sz == length(attributes)
	mask = trues(sz)
	local vcountt
	local vcountp
	local or2t
	local or2p
	Suppressor.@suppress vcountt, vcountp, or2t, or2p = analysis_eachtime(Xon, Xin, times, keepcases; kw...)
	for i = 1:sz
		mask[i] = false
		local vr2t
		local vr2p
		Suppressor.@suppress vcountt, vcountp, vr2t, vr2p = analysis_eachtime(Xon, Xin, times, keepcases; mask=mask, kw...)
		mask[i] = true
		ta = abs.(or2t .- vr2t)
		pa = abs.(or2p .- vr2p)
		te = sum(ta)
		pe = sum(pa)
		@info "$(attributes[i]): $te : $pe"
		# display([ta pa])
	end
end

function analysis_eachtime(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector, keepcases::BitArray; modeltype::Symbol=:svr, ptimes::AbstractUnitRange=1:length(times), plot::Bool=false, trainingrange::AbstractVector=[0.0, 0.05, 0.1, 0.2, 0.33], epsilon::Float64=0.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	for r in trainingrange
		Xe = copy(Xon)
		Printf.@printf("%8s %8s %8s %8s %10s %8s\n", "Ratio", "#Train", "#Pred", "Time", "R2 train", "R2 pred")
		for i = 1:ntimes
			is = sortperm(Xon[:, i])
			T = setdata(i, Xin; mask=mask, order=is)
			if isnothing(T)
				return
			end
			Xen, _, _ = NMFk.normalize(Xe; amin=0)
			if i > 1 # add past estimates or observations for training
				# T = [T Xon[is,1:i-1]] # add past observations
				T = [T Xen[is, 1:i-1]] # add past estimates
			end
			Xe[is, i] .= 0
			local countt = 0
			local countp = 0
			local r2t = 0
			local r2p = 0
			local pm
			tr = i in ptimes ? r : 0
			for k = 1:nreruns
				if modeltype == :svr
					y_pr, pm, _ = svrmodel(Xon[is, i], permutedims(T); ratio=tr, keepcases=keepcases[is])
				else
					@warn("Unknown model type! SVR will be used!")
					y_pr, pm, _ = svrmodel(Xon[is, i], permutedims(T); ratio=tr, keepcases=keepcases[is])
				end
				countt += sum(.!pm)
				countp += sum(pm)
				Xe[is, i] .+= y_pr
				r2 = SVR.r2(Xon[is, i][.!pm], y_pr[.!pm])
				r2t += r2
				if plot
					Mads.plotseries([Xon[is, i] y_pr]; xmin=1, xmax=length(y_pr), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xon[is, i][.!pm], y_pr[.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
				end
				if sum(pm) > 0
					r2 = SVR.r2(Xon[is, i][pm], y_pr[pm])
					r2p += r2
					if plot
						NMFk.plotscatter(Xon[is, i][pm], y_pr[pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
					end
				end
			end
			Xe[is, i] ./= nreruns
			r2 = SVR.r2(Xon[is, i][.!pm], Xe[is, i][.!pm])
			if plot
				Mads.plotseries([Xon[is, i] Xe[is, i]]; xmin=1, xmax=size(Xon[:, i], 1), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(Xon[is, i][.!pm], Xe[is, i][.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if countp > 0
				r2 = SVR.r2(Xon[is, i][pm], Xe[is, i][pm])
				plot && NMFk.plotscatter(Xon[is, i][pm], Xe[is, i][pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if sum(pm) > 0
				Printf.@printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8.4f\n", r, countt / nreruns, countp / nreruns, times[i], r2t / nreruns, r2p / nreruns)
			else
				Printf.@printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8s\n", r, countt / nreruns, countp / nreruns, times[i], r2t / nreruns, "-")
			end
			push!(vcountt, countt / nreruns)
			push!(vcountp, countp / nreruns)
			push!(vr2t, r2t / nreruns)
			push!(vr2p, r2p / nreruns)
		end
		println()
	end
	return vcountt, vcountp, vr2t, vr2p
end

function analysis_transient(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector, keepcases::BitArray, Xtn::AbstractMatrix=Matrix(undef, 0, 0); modeltype::Symbol=:svr, ptimes::Union{Vector{Integer},AbstractUnitRange}=1:length(times), plot::Bool=false, plottime::Bool=plot, trainingrange::AbstractVector=[0.0, 0.05, 0.1, 0.2, 0.33], epsilon::Float64=0.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	vr2tt = Vector{Float64}(undef, ntimes)
	vr2tp = Vector{Float64}(undef, ntimes)
	for r in trainingrange
		if sizeof(Xtn) > 0
			T = setdata(Xin, [times Xtn]; mask=mask)
		else
			T = setdata(Xin, times; mask=mask)
		end
		if isnothing(T)
			return
		end
		local countt = 0
		local countp = 0
		local r2t = 0
		local r2p = 0
		local pm
		vr2tt .= 0
		vr2tp .= 0
		vy_tr = vec(Xon[:, 1:ntimes])
		for k = 1:nreruns
			pm, lpm = setup_mask(r, keepcases, ncases, ntimes, ptimes)
			if modeltype == :svr
				vy_pr, _, _ = svrmodel(vy_tr, permutedims(T); pm=lpm)
			else
				@warn("Unknown model type! SVR will be used!")
				vy_pr, _, _ = svrmodel(vy_tr, permutedims(T); pm=lpm)
			end
			countt += sum(.!pm)
			countp += sum(pm)
			r2 = SVR.r2(vy_tr[.!lpm], vy_pr[.!lpm])
			r2t += r2
			if plot
				Mads.plotseries([vy_tr vy_pr]; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(vy_tr[.!lpm], vy_pr[.!lpm]; title="Training Size: $(sum(.!pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if sum(pm) > 0
				r2 = SVR.r2(vy_tr[lpm], vy_pr[lpm])
				r2p += r2
				plot && NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Prediction Size: $(sum(pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			y_pr = reshape(vy_pr, ncases, ntimes)
			for i = 1:ntimes
				opm = (i in ptimes) ? pm : falses(length(pm))
				r2tt = SVR.r2(Xon[.!opm, i], y_pr[.!opm, i])
				vr2tt[i] += r2tt
				if plottime
					Mads.plotseries([Xon[:, i] y_pr[:, i]]; xmin=1, xmax=size(Xon[:, i], 1), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xon[.!opm, i], y_pr[.!opm, i]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2tt; sigdigits=2))")
				end
				if i in ptimes
					r2tp = SVR.r2(Xon[opm, i], y_pr[opm, i])
					vr2tp[i] += r2tp
					plottime && NMFk.plotscatter(Xon[opm, i], y_pr[opm, i]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2tp; sigdigits=2))")
				end
			end
		end
		Printf.@printf("%8s %8s %8s %8s %10s %8s\n", "Ratio", "#Train", "#Pred", "Time", "R2 train", "R2 pred")
		for i = 1:ntimes
			if i in ptimes
				Printf.@printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8.4f\n", r, countt / nreruns, countp / nreruns, times[i], vr2tt[i] / nreruns, vr2tp[i] / nreruns)
			else
				Printf.@printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8s\n", r, countt / nreruns, countp / nreruns, times[i], vr2tt[i] / nreruns, "-")
			end
		end
		println()
		push!(vcountt, countt / nreruns)
		push!(vcountp, countp / nreruns)
		push!(vr2t, r2t / nreruns)
		push!(vr2p, r2p / nreruns)
	end
	return vcountt, vcountp, vr2t, vr2p
end