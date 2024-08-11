function sensitivity(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector=["Time_$(i)" for i in axes(Xon, 2)], keepcases::BitArray=falses(size(Xon, 1)), attributes::AbstractVector=["Param_$(i)" for i in axes(Xin, 2)]; kw...)
	sz = length(attributes)
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

function analysis_eachtime(Xon::AbstractMatrix, Xin::AbstractMatrix, times::AbstractVector, keepcases::BitArray=falses(size(Xon, 1)); modeltype::Symbol=:svr, ptimes::AbstractUnitRange=1:length(times), plot::Bool=false, trainingrange::AbstractVector=[0.0, 0.05, 0.1, 0.2, 0.33], epsilon::Float64=0.000000001, gamma::Float64=0.1, nreruns::Int64=10, mask=Colon())
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