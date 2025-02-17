function model_fully_transient(Xo::AbstractMatrix, Xi::AbstractMatrix, times::AbstractVector=Vector(undef, 0), Xtn::AbstractMatrix=Matrix(undef, 0, 0); keepcases::BitArray=falses(size(Xo, 1)), modeltype::Symbol=:svr, ratio::Number=0, ptimes::Union{Vector{Integer},AbstractUnitRange}=eachindex(times), plot::Bool=false, plottime::Bool=false, mask=Colon(), load::Bool=false, save::Bool=false, modeldir::AbstractString=joinpath(workdir, "model_$(modeltype)"), plotdir::AbstractString=joinpath(workdir, "figures_$(modeltype)"), case::AbstractString="", filename::AbstractString="", xtitle::Union{AbstractString,Nothing}=:nothing, ytitle::Union{AbstractString,Nothing}=:nothing, quiet::Bool=false, kw...)
	Mads.recursivemkdir(plotdir; filename=false)
	inan = vec(.!isnan.(sum(Xo; dims=2))) .|| vec(.!isnan.(sum(Xi; dims=2)))
	Xon, Xomin, Xomax = NMFk.normalize(Xo[inan, :])
	Xin, Ximin, Ximax = NMFk.normalize(Xi[inan, :])
	ntimes = length(times)
	ncases = size(Xin, 1)
	nparams = size(Xin, 2)
	nparams1 = nparams + 1
	nobs = size(Xon, 2)
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
		@info("Number of cases for validation: $(sum(pm))")
		if ntimes > 0
			@info("Number of cases/transients for training: $(ncases * ntimes - sum(lpm))")
			@info("Number of cases/transients for validation: $(sum(lpm))")
		end
	end
	if plot && ntimes > 0
		Mads.plotseries(Xo[.!pm, 1:ntimes]', "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_training_series.png"; title="Training set ($(sum(.!pm)))", xaxis=times, xmin=0, xmax=times[end], xtitle=xtitle, ytitle=ytitle)
		if sum(pm) > 0
			Mads.plotseries(Xo[pm, 1:ntimes]', "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_validation_series.png"; title="Validation set ($(sum(pm)))", xaxis=times, xmin=0, xmax=times[end], xtitle=xtitle, ytitle=ytitle)
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
	elseif modeltype == :mlj
		vy_prn, _, model = mljmodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet)
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
	function mlmodel(x::AbstractVector)
		xn = first(NMFk.normalize(x; amin=Ximin, amax=Ximax))
		if ntimes > 0
			y = Vector{Float64}(undef, ntimes)
			for i = eachindex(tn)
				y[i] = first(predict(model, reshape([tn[i]; xn], (1, nparams1))))
			end
		else
			y = predict(model, xn)
		end
		NMFk.denormalize!(y, Xomin, Xomax)
		return y
	end
	function mlmodel(t::AbstractVector, x::AbstractVector)
		tnl, _, _ = NMFk.normalize(Float64.(t); amin=tmin, amax=tmax)
		xn = first(NMFk.normalize(x; amin=Ximin, amax=Ximax))
		ntimes = length(t)
		y = Vector{Float64}(undef, ntimes)
		for i = eachindex(tnl)
			y[i] = first(predict(model, reshape([tnl[i]; xn], (1, nparams1))))
		end
		NMFk.denormalize!(y, Xomin, Xomax)
		return y
	end
	function mlmodel(X::AbstractMatrix)
		nc = size(X, 1)
		Xn, _, _ = NMFk.normalize(X; amin=Ximin, amax=Ximax)
		if ntimes > 0
			Yn = Matrix{Float64}(undef, nc, ntimes)
			for i = eachindex(tn)
				S = [repeat(tn[i:i], nc) Xn]
				Yn[:, i] .= predict(model, S)
			end
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
		else
			Yn = predict(model, permutedims(Xn))
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
			Y = reshape(Y, nc, 1)
		end
		return Y
	end
	function mlmodel(t::AbstractVector, X::AbstractMatrix)
		nc = size(X, 1)
		tnl, _, _ = NMFk.normalize(Float64.(t); amin=tmin, amax=tmax)
		ntimes = length(t)
		Xn, _, _ = NMFk.normalize(X; amin=Ximin, amax=Ximax)
		Yn = Matrix{Float64}(undef, nc, ntimes)
		for i = eachindex(tnl)
			S = [repeat(tnl[i:i], nc) Xn]
			Yn[:, i] = predict(model, S)
		end
		Y = NMFk.denormalize(Yn, Xomin, Xomax)
		return Y
	end
	r2t = SVR.r2(vy_tr[.!lpm], vy_pr[.!lpm])
	rmset = NMFk.rmsenan(vy_tr[.!lpm], vy_pr[.!lpm])
	!quiet && println("Training $(ncases - sum(pm)) / $(sum(pm)) $(ratio) r2 = $(round(r2t; sigdigits=3)) rmse = $(round(rmset; sigdigits=3))")
	if sum(lpm) > 0
		r2p = SVR.r2(vy_tr[lpm], vy_pr[lpm])
		rmsep = NMFk.rmsenan(vy_tr[lpm], vy_pr[lpm])
		!quiet && println("Validation $(ncases - sum(pm)) / $(sum(pm)) $(ratio) r2 = $(round(r2p; sigdigits=3)) rmse = $(round(rmsep; sigdigits=3))")
	else
		r2p = rmsep = NaN
	end
	if quiet
		println("$(ncases - sum(pm)) / $(sum(pm)) $(ratio) $(r2t) $(r2p) $(rmset) $(rmsep)")
	end
	if plot
		Mads.plotseries([vy_tr vy_pr], "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_series.png"; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
		NMFk.plotscatter(vy_tr[.!lpm], vy_pr[.!lpm]; title="Training Size: $(sum(.!pm)); r<sup>2</sup>: $(round(r2t; sigdigits=3)); RMSE: $(round(rmset; sigdigits=3))", xtitle="Truth", ytitle="Prediction", filename="$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_scatter_training.png")

	end
	if plot && sum(lpm) > 0
		NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Validation Size: $(sum(pm)); r<sup>2</sup>: $(round(r2p; sigdigits=3)); RMSE: $(round(rmsep; sigdigits=3))", xtitle="Truth", ytitle="Prediction", filename="$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_scatter_validation.png")
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
	return mlmodel, model, y_pr, pm
end

function model(Xo::AbstractMatrix, Xi::AbstractMatrix, times::AbstractVector=Vector(undef, 0), Xtn::AbstractMatrix=Matrix(undef, 0, 0); keepcases::BitArray=falses(size(Xo, 1)), modeltype::Symbol=:svr, ratio::Number=0, ptimes::Union{Vector{Integer},AbstractUnitRange}=eachindex(times), plot::Bool=false, plottime::Bool=false, mask=Colon(), load::Bool=false, save::Bool=false, modeldir::AbstractString=joinpath(workdir, "model_$(modeltype)"), plotdir::AbstractString=joinpath(workdir, "figures_$(modeltype)"), case::AbstractString="", filename::AbstractString="", quiet::Bool=false, xtitle::Union{AbstractString,Nothing}=:nothing, ytitle::Union{AbstractString,Nothing}=:nothing, kw...)
	Mads.recursivemkdir(plotdir; filename=false)
	inan = vec(.!isnan.(sum(Xo; dims=2))) .|| vec(.!isnan.(sum(Xi; dims=2)))
	Xon, Xomin, Xomax, Xob = NMFk.normalizematrix_col(Xo[inan, :])
	Xin, Ximin, Ximax, Xib = NMFk.normalizematrix_col(Xi[inan, :])
	ntimes = length(times)
	ncases = size(Xin, 1)
	nparams = size(Xin, 2)
	nparams1 = nparams + 1
	nobs = size(Xon, 2)
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
		@info("Number of cases for validation: $(sum(pm))")
		if ntimes > 0
			@info("Number of cases/transients for training: $(ncases * ntimes - sum(lpm))")
			@info("Number of cases/transients for validation: $(sum(lpm))")
		end
	end
	if plot && ntimes > 0
		Mads.plotseries(Xo[.!pm, 1:ntimes]', "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_training_series.png"; title="Training set ($(sum(.!pm)))", xaxis=times, xmin=0, xmax=times[end], xtitle=xtitle, ytitle=ytitle)
		if sum(pm) > 0
			Mads.plotseries(Xo[pm, 1:ntimes]', "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_validation_series.png"; title="Validation set ($(sum(pm)))", xaxis=times, xmin=0, xmax=times[end], xtitle=xtitle, ytitle=ytitle)
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
	elseif modeltype == :mlj
		vy_prn, _, model = mljmodel(vy_trn, T; load=load, save=save, filename=filename, pm=lpm, quiet=quiet)
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
	if ntimes > 0
		Xomin_interpolated = Interpolations.CubicSplineInterpolation(0:ntimes-1, aomin, extrapolation_bc=Interpolations.Line())
		Xomax_interpolated = Interpolations.CubicSplineInterpolation(0:ntimes-1, aomax, extrapolation_bc=Interpolations.Line())
	end
	function mlmodel(x::AbstractVector)
		xn = first(NMFk.normalize(x; amin=aimin, amax=aimax))
		if ntimes > 0
			y = Vector{Float64}(undef, ntimes)
			for i = eachindex(tn)
				y[i] = first(predict(model, reshape([tn[i]; xn], (1, nparams1))))
			end
		else
			y = predict(model, reshape(xn, (1, nparams)))
		end
		NMFk.denormalize!(y, aomin, aomax)
		return y
	end
	function mlmodel(t::AbstractVector, x::AbstractVector)
		tnl, _, _ = NMFk.normalize(Float64.(t); amin=tmin, amax=tmax)
		xn = first(NMFk.normalize(x; amin=aimin, amax=aimax))
		ntimes = length(t)
		y = Vector{Float64}(undef, ntimes)
		for i = eachindex(tnl)
			y[i] = first(predict(model, reshape([tnl[i]; xn], (1, nparams1))))
		end
		NMFk.denormalize!(y, Xomin_interpolated(t), Xomax_interpolated(t))
		return y
	end
	function mlmodel(X::AbstractMatrix)
		nc = size(X, 1)
		Xn, _, _, _ = NMFk.normalizematrix_col(X; amin=Ximin, amax=Ximax)
		if ntimes > 0
			Yn = Matrix{Float64}(undef, nc, ntimes)
			for i = eachindex(tn)
				S = [repeat(tn[i:i], nc) Xn]
				Yn[:, i] .= predict(model, S)
			end
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
		else
			Yn = predict(model, Xn)
			Y = NMFk.denormalize(Yn, Xomin, Xomax)
			Y = reshape(Y, (nc, 1))
		end
		return Y
	end
	function mlmodel(t::AbstractVector, X::AbstractMatrix)
		nc = size(X, 1)
		tnl, _, _ = NMFk.normalize(Float64.(t); amin=tmin, amax=tmax)
		ntimes = length(t)
		Xn, _, _, _ = NMFk.normalizematrix_col(X; amin=Ximin, amax=Ximax)
		Yn = Matrix{Float64}(undef, nc, ntimes)
		for i = eachindex(tnl)
			S = [repeat(tnl[i:i], nc) Xn]
			Yn[:, i] = predict(model, S)
		end
		Y = NMFk.denormalize(Yn, reshape(Xomin_interpolated(t), (1, ntimes)), reshape(Xomax_interpolated(t), (1, ntimes)))
		return Y
	end
	r2t = SVR.r2(vy_tr[.!lpm], vy_pr[.!lpm])
	rmset = NMFk.rmsenan(vy_tr[.!lpm], vy_pr[.!lpm])
	!quiet && println("Training $(ncases - sum(pm)) / $(sum(pm)) $(ratio) r2 = $(round(r2t; sigdigits=3)) rmse = $(round(rmset; sigdigits=3))")
	if sum(lpm) > 0
		r2p = SVR.r2(vy_tr[lpm], vy_pr[lpm])
		rmsep = NMFk.rmsenan(vy_tr[lpm], vy_pr[lpm])
		!quiet && println("Validation $(ncases - sum(pm)) / $(sum(pm)) $(ratio) r2 = $(round(r2p; sigdigits=3)) rmse = $(round(rmsep; sigdigits=3))")
	else
		r2p = rmsep = NaN
	end
	if quiet
		println("$(ncases - sum(pm)) / $(sum(pm)) $(ratio) $(r2t) $(r2p) $(rmset) $(rmsep)")
	end
	if plot
		Mads.plotseries([vy_tr vy_pr], "$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_series.png"; xmin=1, xmax=length(vy_pr), logy=false, names=["Truth", "Prediction"])
		NMFk.plotscatter(vy_tr[.!lpm], vy_pr[.!lpm]; title="Training Size: $(sum(.!pm)); r<sup>2</sup>: $(round(r2t; sigdigits=3)); RMSE: $(round(rmset; sigdigits=3))", xtitle="Truth", ytitle="Prediction", filename="$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_scatter_training.png")

	end
	if plot && sum(lpm) > 0
		NMFk.plotscatter(vy_tr[lpm], vy_pr[lpm]; title="Validation Size: $(sum(pm)); r<sup>2</sup>: $(round(r2p; sigdigits=3)); RMSE: $(round(rmsep; sigdigits=3))", xtitle="Truth", ytitle="Prediction", filename="$(plotdir)/$(case)_$(ncases)_$(ncases - sum(pm))_$(sum(pm))_scatter_validation.png")
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
	return mlmodel, model, y_pr, pm
end