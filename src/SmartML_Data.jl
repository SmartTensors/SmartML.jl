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

function setfilename(filename::AbstractString, dirname::AbstractString, prefix::AbstractString, suffix::AbstractString)
	filename = filename == "" ? prefix * suffix : filename
	Mads.recursivemkdir(joinpath(dirname, filename); filename=true)
	return filename
end