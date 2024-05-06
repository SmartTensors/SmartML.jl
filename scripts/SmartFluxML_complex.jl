import Flux
import CUDA
import Mads
import Random
import Gadfly

workdir = @show(@__DIR__)

# Random.seed!(2022)
# r = rand(100)
f1(x; v=1:100) = (sin.(v ./ (0.5 + 10 * x)) ./ 2 .+ 0.5) ./ 5
s1 = f1(1)
f2(x; v=1:100) = (exp.(v ./(5 + x)) .- 1 ) ./ 6e8
s2 = f2(0)
f3(x; v=1:100) = (sin.(v ./ (0.5 + x)) ./ 2 .+ 0.5) ./ 5
s3 = f3(1)

Mads.plotseries([s1 s2 s3], joinpath(workdir, "figures", "Complex Original Equations.png"); hsize=12Gadfly.inch, title="Original Equations", name="Equation", xtitle="Time", ytitle="Value")

y = s1 .+ s2 .+ s3

Mads.plotseries(y, joinpath(workdir, "figures", "Complex Mixed Equations.png"); title="Mixed Equations", name="", xtitle="Time", ytitle="Value")

function anal_model_c(p; v=1:100)
	y = [f1(p[1]; v=v) f2(p[2]; v=v) f3(p[3]; v=v)]
	y = vec(sum(y; dims=2))
	return y
end

Mads.plotseries(anal_model_c(rand(3)), joinpath(workdir, "figures", "Complex Mixed Equations.png"); title="Mixed Equations", name="", xtitle="Time", ytitle="Value")

function ml0_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.Dense(input, 32), Flux.Dense(32, output)))
	return model, Flux.params(model)
end

function ml1_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Dense(input, 32), Flux.Dense(32, output)), (x2, x)->(x2 .+ hcat(f1.(vec(x[1:1,:]); v=v)...)))))
	return model, Flux.params(model)
end

function ml2_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Dense(input, 32), Flux.Dense(32, output)), (x2, x)->(x2 .+ hcat(f2.(vec(x[2:2,:]); v=v)...)))))
	return model, Flux.params(model)
end

function ml3_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Dense(input, 32), Flux.Dense(32, output)), (x2, x)->(x2 .+ hcat(f3.(vec(x[3:3,:]); v=v)...)))))
	return model, Flux.params(model)
end

function ml123_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Dense(input, 32), Flux.Dense(32, output)), (x2, x)->(x2 .+ hcat(f1.(vec(x[1:1,:]); v=v)...) .+ hcat(f2.(vec(x[2:2,:]); v=v)...) .+ hcat(f3.(vec(x[3:3,:]); v=v)...)))))
	return model, Flux.params(model)
end

function ml1a_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	d1 = device(Flux.Dense(input-1, 32))
	d2 = device(Flux.Dense(32, output))
	model = device(Flux.Chain(x->(x, d1(x[1:end-1,:])), ((x, x1)::Tuple)->(x, d2(x1)), ((x, x2)::Tuple)->(x2 .+ hcat(f1.(vec(x[1:1,:]); v=v)...))))
	return model, Flux.params((d1, d2))
end

function ml2a_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	d1 = device(Flux.Dense(input-1, 32))
	d2 = device(Flux.Dense(32, output))
	model = device(Flux.Chain(x->(x, d1(x[1:end-1,:])), ((x, x1)::Tuple)->(x, d2(x1)), ((x, x2)::Tuple)->(x2 .+ hcat(f2.(vec(x[2:2,:]); v=v)...))))
	return model, Flux.params((d1, d2))
end

function ml3a_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	d1 = device(Flux.Dense(input-1, 32))
	d2 = device(Flux.Dense(32, output))
	model = device(Flux.Chain(x->(x, d1(x[1:end-1,:])), ((x, x1)::Tuple)->(x, d2(x1)), ((x, x2)::Tuple)->(x2 .+ hcat(f3.(vec(x[3:3,:]); v=v)...))))
	return model, Flux.params((d1, d2))
end

function ml123a_model_c(; input=3, output=100, v=1:output, device=Flux.gpu)
	d1 = device(Flux.Dense(input-1, 32))
	d2 = device(Flux.Dense(32, output))
	model = device(Flux.Chain(x->(x, d1(x[1:end-1,:])), ((x, x1)::Tuple)->(x, d2(x1)), ((x, x2)::Tuple)->(x2 .+ hcat(f1.(vec(x[1:1,:]); v=v)...) .+ hcat(f2.(vec(x[2:2,:]); v=v)...) .+ hcat(f3.(vec(x[3:3,:]); v=v)...))))
	return model, Flux.params((d1, d2))
end

function getdata(args)
	xtrain = rand(3, args.sizetrain)
	ytrain = hcat([anal_model_c(xtrain[:,i]) for i=1:args.sizetrain]...)
	Mads.plotseries(ytrain, joinpath(workdir, "figures", "Complex Training Data.png"); title="Training Data", name="", xtitle="Time", ytitle="Value")
	xtest = rand(3, args.sizetest)
	ytest = hcat([anal_model_c(xtest[:,i]) for i=1:args.sizetest]...)
	Mads.plotseries(ytest, joinpath(workdir, "figures", "Complex Testing Data.png"); title="Testing Data", name="", xtitle="Time", ytitle="Value")

	train_loader = Flux.Data.DataLoader((xtrain, ytrain); batchsize=args.batchsize, shuffle=true)
	test_loader = Flux.Data.DataLoader((xtest, ytest); batchsize=args.batchsize)

	return train_loader, test_loader
end

function loss(data_loader, model, device)
	ls = 0.0f0
	num = 0
	for (x, y) in data_loader
		x, y = device(x), device(y)
		ŷ = model(x)
		ls += Flux.Losses.mse(ŷ, y, agg=sum)
		num += size(x)[end]
	end
	return ls / num
end

Base.@kwdef mutable struct Args
	sizetrain = 500
	sizetest = 500
	η::Float64 = 3e-4        # learning rate
	batchsize::Int = 100     # batch size
	epochs::Int = 1000       # number of epochs
	use_cuda::Bool = false   # use gpu (if cuda available)
end

function train(ml_model; kws...)
	args = Args(; kws...) # collect options in a struct for convenience

	if CUDA.functional() && args.use_cuda
		@info "Training on CUDA GPU"
		CUDA.allowscalar(false)
		device = Flux.gpu
	else
		@info "Training on CPU"
		device = Flux.cpu
	end

	# Construct model
	v = collect(1:100)
	v = device(v)
	model, params = ml_model(; v=v, device=device)

	train_loader, test_loader = getdata(args)

	## Optimizer
	opt = Flux.ADAM(args.η)

	## Training
	v_train_loss = Vector{Float32}(undef, args.epochs)
	v_test_loss = Vector{Float32}(undef, args.epochs)
	for epoch in 1:args.epochs
		for (x, y) in train_loader
			x, y = device(x), device(y) # transfer data to device
			graidents = Flux.gradient(()->Flux.Losses.mse(model(x), y), params) # compute gradient
			Flux.Optimise.update!(opt, params, graidents) # update parameters
		end
		train_loss = loss(train_loader, model, device)
		test_loss = loss(test_loader, model, device)
		if epoch % 20 == 0
			println("Epoch = $epoch")
			println("  train loss = $train_loss")
			println("  test loss  = $test_loss")
		end
		v_train_loss[epoch] = train_loss
		v_test_loss[epoch] = test_loss
	end
	train_loss = loss(train_loader, model, device)
	test_loss = loss(test_loader, model, device)
	println("Final after $(args.epochs) epochs")
	println("  train loss = $train_loss")
	println("  test loss  = $test_loss")

	return model, v_train_loss, v_test_loss
end

model_c0, train_loss_c0, test_loss_c0 = train(ml0_model_c)
model_c1, train_loss_c1, test_loss_c1 = train(ml1_model_c)
model_c2, train_loss_c2, test_loss_c2 = train(ml2_model_c)
model_c3, train_loss_c3, test_loss_c3 = train(ml3_model_c)
model_c123, train_loss_c123, test_loss_c123 = train(ml123_model_c)
p = rand(3)
Mads.plotseries([anal_model_c(p) model_c0(p) model_c1(p) model_c2(p) model_c3(p) model_c123(p)], joinpath(workdir, "figures", "Complex Model Comparisons.png"); names=["Truth", "ML", "PIML 1", "PIML 2", "PIML 3", "PIML 123"], hsize=12Gadfly.inch, xtitle="Time", ytitle="Value", truth=true)
Mads.plotseries([anal_model_c(p) model_c0(p)], joinpath(workdir, "figures", "Complex Model Comparisons ML.png"); names=["Truth", "ML"], hsize=12Gadfly.inch, xtitle="Time", ytitle="Value", truth=true)
Mads.plotseries([anal_model_c(p) model_c0(p) model_c123(p)], joinpath(workdir, "figures", "Complex Model Comparisons ML PIML.png"); names=["Truth", "ML", "PIML"], hsize=12Gadfly.inch, xtitle="Time", ytitle="Value", truth=true)
Mads.plotseries([anal_model_c(p) model_c0(p) model_c3(p)], joinpath(workdir, "figures", "Complex Model Comparisons ML PIML 2.png"); names=["Truth", "ML", "PIML"], hsize=12Gadfly.inch, xtitle="Time", ytitle="Value", truth=true)
Mads.plotseries([train_loss_c0 train_loss_c1 train_loss_c2 train_loss_c3 train_loss_c123], joinpath(workdir, "figures", "Complex Model Training.png"); names=["ML", "PIML 1", "PIML 2", "PIML 3", "PIML 123"], hsize=12Gadfly.inch, logy=true, xtitle="Epoch", ytitle="Loss")
Mads.plotseries([train_loss_c0 train_loss_c123], joinpath(workdir, "figures", "Complex Model Training PIML.png"); names=["ML", "PIML"], hsize=12Gadfly.inch, logy=true, xtitle="Epoch", ytitle="Loss")
Mads.plotseries([test_loss_c0 test_loss_c1 test_loss_c2 test_loss_c3 test_loss_c123], joinpath(workdir, "figures", "Complex Model Testing.png"); names=["ML", "PIML 1", "PIML 2", "PIML 3", "PIML 123"], hsize=12Gadfly.inch, logy=true, xtitle="Epoch", ytitle="Loss")
Mads.plotseries([test_loss_c0 test_loss_c123], joinpath(workdir, "figures", "Complex Model Testing PIML.png"); names=["ML", "PIML"], hsize=12Gadfly.inch, logy=true, xtitle="Epoch", ytitle="Loss")

model_c1a, train_loss_c1a, test_loss_c1a = train(ml1a_model_c)
model_c2a, train_loss_c2a, test_loss_c2a = train(ml2a_model_c)
model_c3a, train_loss_c3a, test_loss_c3a = train(ml3a_model_c)
model_c123a, train_loss_c123a, test_loss_c123a = train(ml123a_model_c)
p = rand(3)
Mads.plotseries([anal_model_c(p) model_c0(p) model_c1a(p) model_c2a(p) model_c3a(p) model_c123a(p)], joinpath(workdir, "figures", "Complex Model A Comparisons.png"); names=["Truth", "ML", "PIML 1", "PIML 2", "PIML 3", "PIML 123"], hsize=12Gadfly.inch, xtitle="Time", ytitle="Value", truth=true)
Mads.plotseries([train_loss_c0 train_loss_c1a train_loss_c2a train_loss_c3a train_loss_c123a], joinpath(workdir, "figures", "Complex Model A Training.png"); names=["ML", "PIML 1", "PIML 2", "PIML 3", "PIML 123"], hsize=12Gadfly.inch, logy=true)
Mads.plotseries([test_loss_c0 test_loss_c1a test_loss_c2a test_loss_c3a test_loss_c123a], joinpath(workdir, "figures", "Complex Model A Testing.png"); names=["ML", "PIML 1", "PIML 2", "PIML 3", "PIML 123"], hsize=12Gadfly.inch, logy=true)