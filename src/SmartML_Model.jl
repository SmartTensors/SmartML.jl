import Mads
import NMFk
import SVR
import ScikitLearn
import XGBoost
import PyCall
xgb = PyCall.pyimport("xgboost")
import Printf
import Suppressor

function setdata(Xin::AbstractMatrix, Xt::AbstractMatrix; order=Colon(), mask=Colon())
	ntimes = size(Xt, 1)
	ncases = size(Xin, 1)
	T = [repeat(Xt; inner=(ncases, 1)) repeat(Xin[order,mask], ntimes)]
	@info("Number of training cases: $(ncases)")
	@info("Number of training times: $(ntimes)")
	@info("Number of training cases * times: $(size(T, 1))")
	@info("Number of training parameters: $(size(T, 2))")
	return T
end

function setdata(Xin::AbstractMatrix, times::AbstractVector; order=Colon(), mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	T = [repeat(times; inner=ncases) repeat(Xin[order,mask], ntimes)]
	@info("Number of training cases: $(ncases)")
	@info("Number of training times: $(ntimes)")
	@info("Number of training cases * times: $(size(T, 1))")
	@info("Number of training parameters: $(size(T, 2))")
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

function fluxmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0., keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, check::Bool=false, load::Bool=false, save::Bool=false, filemodel::AbstractString, kw...)
end

function xgbmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0., keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, load::Bool=false, save::Bool=false, filemodel::AbstractString, kw...)
	if pm === nothing
		pm = SVR.get_prediction_mask(length(y), ratio; keepcases=keepcases, debug=true)
	else
		@assert length(pm) == size(x, 1)
		@assert eltype(pm) <: Bool
	end
	if load && isfile(filemodel)
		@info("Loading model from file: $(filemodel)")
		m = SVR.loadmodel(filemodel)
	else
		@info("Training ...")
		@show size(y)
		@show size(y[.!pm])
		m = XGBoost.xgboost(x[.!pm,:], 20; label=y[.!pm], verbose=0)
		if save && filemodel != ""
			@info("Saving model to file: $(filemodel)")
			Mads.recursivemkdir(filemodel; filename=true)
			XGBoost.save(filemodel, m)
		end
	end
	y_pr = XGBoost.predict(m, x)
	return y_pr, pm, m
end

function xgbtmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0., keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, load::Bool=false, save::Bool=false, filemodel::AbstractString, kw...)
	if pm === nothing
		pm = SVR.get_prediction_mask(length(y), ratio; keepcases=keepcases, debug=true)
	else
		@assert length(pm) == size(x, 1)
		@assert eltype(pm) <: Bool
	end
	if load && isfile(filemodel)
		@info("Loading model from file: $(filemodel)")
		xgb_model = SVR.loadmodel(filemodel)
	else
		@info("Training ...")
		mod = xgb.XGBRegressor(seed = 20)
		param_dict = Dict("max_depth"=>[3, 5, 6, 10, 15, 20],
			"learning_rate"=>[0.01, 0.1, 0.2, 0.3],
			"subsample"=>collect(0.5:0.1:1.0),
			"colsample_bytree"=>collect(0.4:0.1:1.0),
			"colsample_bylevel"=>collect(0.4:0.1:1.0),
			"n_estimators"=>[100, 500, 1000])
		model = ScikitLearn.GridSearch.RandomizedSearchCV(mod, param_dict; verbose=1, n_jobs=1, n_iter=10, cv=5)
		ScikitLearn.fit!(model, x[.!pm,:], y[.!pm])
		xgb_model = model.best_estimator_
		xgb_model.fit(x[.!pm,:], y[.!pm])
		if save && filemodel != ""
			@info("Saving model to file: $(filemodel)")
			Mads.recursivemkdir(filemodel; filename=true)
			XGBoost.save(filemodel, xgb_model)
		end
	end
	y_pr = xgb_model.predict(x)
	return y_pr, pm, xgb_model
end

function svrmodel(y::AbstractVector, x::AbstractMatrix; ratio::Number=0., keepcases::BitArray=trues(length(y)), pm::Union{AbstractVector,Nothing}=nothing, normalize::Bool=true, scale::Bool=true, epsilon::Float64=.000000001, gamma::Float64=0.1, check::Bool=false, load::Bool=false, save::Bool=false, filemodel::AbstractString, kw...)
	if pm === nothing
		pm = SVR.get_prediction_mask(length(y), ratio; keepcases=keepcases, debug=true)
	else
		@assert length(pm) == size(x, 1)
		@assert eltype(pm) <: Bool
	end
	xt = permutedims(x)
	if load && isfile(filemodel)
		@info("Loading model from file: $(filemodel)")
		m = SVR.loadmodel(filemodel)
	else
		@info("Training ...")
		m = SVR.train(y[.!pm], xt[:,.!pm]; epsilon=epsilon, gamma=gamma)
		if save && filemodel != ""
			@info("Saving model to file: $(filemodel)")
			Mads.recursivemkdir(filemodel; filename=true)
			SVR.savemodel(m, filemodel)
		end
	end
	y_pr = SVR.predict(m, xt)
	if check
		y_pr2, _, _ = SVR.fit_test(y, xt; ratio=ratio, quiet=true, pm=pm, keepcases=keepcases, epsilon=epsilon, gamma=gamma, kw...)
		@assert vy_pr == vy_pr2
	end
	return y_pr, pm, m
end

function model(Xo::AbstractMatrix, Xi::AbstractMatrix, times::AbstractVector=Vector(undef, 0), Xtn::AbstractMatrix=Matrix(undef, 0, 0); keepcases::BitArray=falses(size(Xo, 1)), modeltype::Symbol=:svr, ratio::Number=0, ptimes::Union{Vector{Integer},AbstractUnitRange}=1:length(times), plot::Bool=false, plottime::Bool=false, mask=Colon(), load::Bool=false, save::Bool=false, modeldir::AbstractString=joinpath(dir, "$(modeltype)"), case::AbstractString="", filemodel::AbstractString="", kw...)
	inan = vec(.!isnan.(sum(Xo; dims=2))) .|| vec(.!isnan.(sum(Xi; dims=2)))
	Xon, Xomin, Xomax, Xob = NMFk.normalizematrix_col(Xo[inan,:])
	Xin, Ximin, Ximax, Xib = NMFk.normalizematrix_col(Xi[inan,:])
	ntimes = length(times)
	ncases = size(Xin, 1)
	if sizeof(Xtn) > 0
		@assert size(Xtn, 1) == ntimes
		tn, tmin, tmax = NMFk.normalize(Float64.(times))
		T = setdata(Xin, [tn Xtn]; mask=mask)
	elseif ntimes > 0
		tn, tmin, tmax = NMFk.normalize(Float64.(times))
		T = setdata(Xin, tn; mask=mask)
	else
		@assert size(Xon, 2) == 1
		T = Xin[mask,:]
	end
	if ntimes > 0
		vy_trn = vec(Xon[:,1:ntimes])
		pm, lpm = setup_mask(ratio, keepcases, ncases, ntimes, ptimes)
	else
		vy_trn = vec(Xon)
		pm = SVR.get_prediction_mask(length(vy_trn), ratio; keepcases=keepcases, debug=false)
		lpm = pm
	end
	@info("Number of cases for training: $(ncases - sum(pm))")
	@info("Number of cases for prediction: $(sum(pm))")
	if ntimes > 0
		@info("Number of cases/transients for training: $(ncases * ntimes - sum(lpm))")
		@info("Number of cases/transients for prediction: $(sum(lpm))")
	end
	if plot && ntimes > 0
		Mads.plotseries(Xo[.!pm,1:ntimes]'; title="Training set")
		if sum(pm) > 0
			Mads.plotseries(Xo[pm,1:ntimes]'; title="Prediction set")
		end
	end
	if (load || save) && (filemodel != "" || case != "")
		filemodel = joinpath(modeldir, "$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm)).$(modeltype)model")
	end
	if modeltype == :svr
		vy_prn, _, m = svrmodel(vy_trn, T; load=load, save=save, filemodel=filemodel, pm=lpm, kw...)
	elseif modeltype == :xgb
		vy_prn, _, m = xgbmodel(vy_trn, T; load=load, save=save, filemodel=filemodel, pm=lpm)
	elseif modeltype == :xgbt
		vy_prn, _, m = xgbtmodel(vy_trn, T; load=load, save=save, filemodel=filemodel, pm=lpm)		
	elseif modeltype == :flux
		vy_prn, _, m = fluxmodel(vy_trn, T; load=load, save=save, filemodel=filemodel, pm=lpm)
	elseif modeltype == :piml
		vy_prn, _, m = pimlmodel(vy_trn, T; load=load, save=save, filemodel=filemodel, pm=lpm)
	else
		@warn("Unknown model type! SVR will be used!")
		vy_prn, _, m = svrmodel(vy_trn, T; load=load, save=save, filemodel=filemodel, pm=lpm, kw...)
	end
	if ntimes > 0
		vy_prn = reshape(vy_prn, ncases, ntimes)
		vy_trn = reshape(vy_trn, ncases, ntimes)
	else
		vy_prn = reshape(vy_prn, ncases, 1)
		vy_trn = reshape(vy_trn, ncases, 1)
	end
	vy_prn[vy_prn .< 0] .= 0
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
				y[t] = first(SVR.predict(m, [tn[t]; xn]))
			end
		else
			y = SVR.predict(m, xn)
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
				Yn[:,t] = SVR.predict(m, S')
			end
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
		else
			Yn = SVR.predict(m, Xn')
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
			Y = reshape(Y, nc, 1)
		end
		return Y
	end
	r2 = SVR.r2(vy_tr[.!lpm], vy_pr[.!lpm])
	if plot
		Mads.plotseries([vy_tr vy_pr]; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
		NMFk.plotscatter(vy_tr[.!lpm], vy_pr[.!lpm]; title="Training Size: $(sum(.!pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
		println("Training r2 = $(round(r2; sigdigits=2))")
	end
	if plot && sum(lpm) > 0
		NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Prediction Size: $(sum(pm)); r<sup>2</sup>: $(round(r2; sigdigits=2))")
		println("Prediction r2 = $(round(r2; sigdigits=2))")
	end
	if ntimes > 0
		y_pr = reshape(vy_pr, ncases, ntimes)
	else
		y_pr = reshape(vy_pr, ncases, 1)
	end
	if plottime && ntimes > 0
		for i = 1:ncases
			Mads.plotseries(permutedims([Xo[i:ncases:end,:]; y_pr[i:ncases:end,:]]); xaxis=times, xmin=0, xmax=times[end], logy=false, names=["Truth", "Prediction"])
		end
	end
	return mlmodel, m, y_pr, T
end

function sensitivity(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector, keepcases::BitArray, attributes::AbstractVector; kw...)
	@assert sz == length(attributes)
	mask = trues(sz)
	local vcountt
	local vcountp
	local or2t
	local or2p
	@Suppressor.suppress vcountt, vcountp, or2t, or2p = analysis_eachtime(Xon, Xin, times, keepcases; kw...)
	for i = 1:sz
		mask[i] = false
		local vr2t
		local vr2p
		@Suppressor.suppress vcountt, vcountp, vr2t, vr2p = analysis_eachtime(Xon, Xin, times, keepcases; mask=mask, kw...)
		mask[i] = true
		ta = abs.(or2t .- vr2t)
		pa = abs.(or2p .- vr2p)
		te = sum(ta)
		pe = sum(pa)
		@info "$(attributes[i]): $te : $pe"
		# display([ta pa])
	end
end

function analysis_eachtime(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector, keepcases::BitArray; modeltype::Symbol=:svr, ptimes::AbstractUnitRange=1:length(times), plot::Bool=false, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
	ntimes = length(times)
	ncases = size(Xin, 1)
	vcountt = Vector{Int64}(undef, 0)
	vcountp = Vector{Int64}(undef, 0)
	vr2t = Vector{Float64}(undef, 0)
	vr2p = Vector{Float64}(undef, 0)
	for r in trainingrange
		Xe = copy(Xon)
		@Printf.printf("%8s %8s %8s %8s %10s %8s\n", "Ratio", "#Train", "#Pred", "Time", "R2 train", "R2 pred")
		for i = 1:ntimes
			is = sortperm(Xon[:,i])
			T = setdata(i, Xin; mask=mask, order=is)
			if isnothing(T)
				return
			end
			Xen, _, _ = NMFk.normalize(Xe; amin=0)
			if i > 1 # add past estimates or observations for training
				# T = [T Xon[is,1:i-1]] # add past observations
				T = [T Xen[is,1:i-1]] # add past estimates
			end
			Xe[is,i] .= 0
			local countt = 0
			local countp = 0
			local r2t = 0
			local r2p = 0
			local pm
			tr = i in ptimes ? r : 0
			for k = 1:nreruns
				if modeltype == :svr
					y_pr, pm, _ = svrmodel(Xon[is,i], permutedims(T); ratio=tr, keepcases=keepcases[is])
				else
					@warn("Unknown model type! SVR will be used!")
					y_pr, pm, _ = svrmodel(Xon[is,i], permutedims(T); ratio=tr, keepcases=keepcases[is])
				end
				countt += sum(.!pm)
				countp += sum(pm)
				Xe[is,i] .+= y_pr
				r2 = SVR.r2(Xon[is,i][.!pm], y_pr[.!pm])
				r2t += r2
				if plot
					Mads.plotseries([Xon[is,i] y_pr]; xmin=1, xmax=length(y_pr), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xon[is,i][.!pm], y_pr[.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
				end
				if sum(pm) > 0
					r2 = SVR.r2(Xon[is,i][pm], y_pr[pm])
					r2p += r2
					if plot
						NMFk.plotscatter(Xon[is,i][pm], y_pr[pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
					end
				end
			end
			Xe[is,i] ./= nreruns
			r2 = SVR.r2(Xon[is,i][.!pm], Xe[is,i][.!pm])
			if plot
				Mads.plotseries([Xon[is,i] Xe[is,i]]; xmin=1, xmax=size(Xon[:,i], 1), logy=false, names=["Truth", "Prediction"])
				NMFk.plotscatter(Xon[is,i][.!pm], Xe[is,i][.!pm]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if countp > 0
				r2 = SVR.r2(Xon[is,i][pm], Xe[is,i][pm])
				plot && NMFk.plotscatter(Xon[is,i][pm], Xe[is,i][pm]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2; sigdigits=2))")
			end
			if sum(pm) > 0
				@Printf.printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8.4f\n", r, countt / nreruns, countp / nreruns, times[i], r2t / nreruns, r2p / nreruns)
			else
				@Printf.printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8s\n", r, countt / nreruns, countp / nreruns, times[i], r2t / nreruns, "-")
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

function analysis_transient(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector, keepcases::BitArray, Xtn::AbstractMatrix=Matrix(undef, 0, 0); modeltype::Symbol=:svr, ptimes::Union{Vector{Integer},AbstractUnitRange}=1:length(times), plot::Bool=false, plottime::Bool=plot, trainingrange::AbstractVector=[0., 0.05, 0.1, 0.2, 0.33], epsilon::Float64=.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
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
		vy_tr = vec(Xon[:,1:ntimes])
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
				r2tt = SVR.r2(Xon[.!opm,i], y_pr[.!opm,i])
				vr2tt[i] += r2tt
				if plottime
					Mads.plotseries([Xon[:,i] y_pr[:,i]]; xmin=1, xmax=size(Xon[:,i], 1), logy=false, names=["Truth", "Prediction"])
					NMFk.plotscatter(Xon[.!opm,i], y_pr[.!opm,i]; title="Training: Time: $(times[i]) days; Count: $(countt); r<sup>2</sup>: $(round(r2tt; sigdigits=2))")
				end
				if i in ptimes
					r2tp = SVR.r2(Xon[opm,i], y_pr[opm,i])
					vr2tp[i] += r2tp
					plottime && NMFk.plotscatter(Xon[opm,i], y_pr[opm,i]; title="Prediction: Time: $(times[i]) days; Count: $(countp); r<sup>2</sup>: $(round(r2tp; sigdigits=2))")
				end
			end
		end
		@Printf.printf("%8s %8s %8s %8s %10s %8s\n", "Ratio", "#Train", "#Pred", "Time", "R2 train", "R2 pred")
		for i = 1:ntimes
			if i in ptimes
				@Printf.printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8.4f\n", r, countt / nreruns, countp / nreruns, times[i], vr2tt[i] / nreruns, vr2tp[i] / nreruns)
			else
				@Printf.printf("%8.2f %8.0f %8.0f %8.0f %10.4f %8s\n", r, countt / nreruns, countp / nreruns, times[i], vr2tt[i] / nreruns, "-")
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