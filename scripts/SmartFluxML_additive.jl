import Flux
import CUDA
import Mads
import Random
import Cairo, Fontconfig

Random.seed!(2022)
r = rand(100)
f1(x) = (sin(x/10) / 2 + 0.5) / 5
s1 = f1.(1:100)
f2(x) = (exp(x/5) - 1 ) / 6e8
s2 = f2.(1:100)
f3(x) = (sin(x) / 2 + 0.5) / 5
s3 = f3.(1:100)

Mads.plotseries([s1 s2 s3], "Original signals.png"; title="Original signals", name="Signal")

y = s1 .+ s2 .+ s3

Mads.plotseries(y, "Mixed signals.png"; title="Mixed signals", name="")

function anal_model(p; v=1:100)
	y = [f1.(v) f2.(v) f3.(v)] .* permutedims(p)
	y = vec(sum(y; dims=2))
	return y
end

Mads.plotseries(anal_model([1,1,1]), "Mixed signals.png"; title="Mixed signals", name="")

function ml0_model(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.Dense(input, 3), Flux.Dense(3, output)))
	return model, Flux.params(model)
end

function ml1_model(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Dense(input, 3), Flux.Dense(3, output)), (x2, x)->(x2 .+ x[1:1,:] .* f1.(v)))))
	return model, Flux.params(model)
end

function ml2_model(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Dense(input, 3), Flux.Dense(3, output)), (x2, x)->(x2 .+ x[2:2,:] .* f2.(v)))))
	return model, Flux.params(model)
end

function ml3_model(; input=3, output=100, v=1:output, device=Flux.gpu)
	model = device(Flux.Chain(Flux.SkipConnection(Flux.Chain(Flux.Dense(input, 3), Flux.Dense(3, output)), (x2, x)->(x2 .+ x[3:3,:] .* f3.(v)))))
	return model, Flux.params(model)
end

function ml1a_model(; input=3, output=100, v=1:output, device=Flux.gpu)
	d1 = device(Flux.Dense(input-1, 3))
	d2 = device(Flux.Dense(3, output))
	model = device(Flux.Chain(x->(x, d1(x[1:end-1,:])), ((x, x1)::Tuple)->(x, d2(x1)), ((x, x2)::Tuple)->(x2 .+ x[1:1,:] .* f1.(v))))
	return model, Flux.params((d1, d2))
end

function ml2a_model(; input=3, output=100, v=1:output, device=Flux.gpu)
	d1 = device(Flux.Dense(input-1, 3))
	d2 = device(Flux.Dense(3, output))
	model = device(Flux.Chain(x->(x, d1(x[1:end-1,:])), ((x, x1)::Tuple)->(x, d2(x1)), ((x, x2)::Tuple)->(x2 .+ x[2:2,:] .* f2.(v))))
	return model, Flux.params((d1, d2))
end

function ml3a_model(; input=3, output=100, v=1:output, device=Flux.gpu)
	d1 = device(Flux.Dense(input-1, 3))
	d2 = device(Flux.Dense(3, output))
	model = device(Flux.Chain(x->(x, d1(x[1:end-1,:])), ((x, x1)::Tuple)->(x, d2(x1)), ((x, x2)::Tuple)->(x2 .+ x[3:3,:] .* f3.(v))))
	return model, Flux.params((d1, d2))
end

function getdata(args)
	xtrain = rand(3, args.sizetrain)
	ytrain = hcat([anal_model(xtrain[:,i]) for i=1:args.sizetrain]...)
	Mads.plotseries(ytrain, "Training data.png"; title="Training data", name="")
	xtest = rand(3, args.sizetest)
	ytest = hcat([anal_model(xtest[:,i]) for i=1:args.sizetest]...)
	Mads.plotseries(ytrain, "Testing data.png"; title="Testing data", name="")

	train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=args.batchsize, shuffle=true)
	test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=args.batchsize)

	return train_loader, test_loader
end

function loss(data_loader, model, device)
	acc = 0
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

@Base.kwdef mutable struct Args
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

model0, train_loss0, test_loss0 = train(ml0_model)
model1, train_loss1, test_loss1 = train(ml1_model)
model2, train_loss2, test_loss2 = train(ml2_model)
model3, train_loss3, test_loss3 = train(ml3_model)
p = rand(3)
Mads.plotseries([anal_model(p) model0(p) model1(p) model2(p) model3(p)], "Model comparisons.png"; names=["Truth", "ML", "PIML 1", "PIML 2", "PIML 3"], hsize=12Gadfly.inch)
Mads.plotseries([train_loss0 train_loss1 train_loss2 train_loss3], "Model training.png"; names=["ML", "PIML 1", "PIML 2", "PIML 3"], hsize=12Gadfly.inch, logy=true)
Mads.plotseries([test_loss0 test_loss1 test_loss2 test_loss3], "Model testing.png"; names=["ML", "PIML 1", "PIML 2", "PIML 3"], hsize=12Gadfly.inch, logy=true)

model1a, train_loss1a, test_loss1a = train(ml1a_model)
model2a, train_loss2a, test_loss2a = train(ml2a_model)
model3a, train_loss3a, test_loss3a = train(ml3a_model)
p = rand(3)
Mads.plotseries([anal_model(p) model0(p) model1a(p) model2a(p) model3a(p)], "Model A comparisons.png"; names=["Truth", "ML", "PIML 1", "PIML 2", "PIML 3"], hsize=12Gadfly.inch)
Mads.plotseries([train_loss0 train_loss1a train_loss2a train_loss3a], "Model A training.png"; names=["ML", "PIML 1", "PIML 2", "PIML 3"], hsize=12Gadfly.inch, logy=true)
Mads.plotseries([test_loss0 test_loss1a test_loss2a test_loss3a], "Model A testing.png"; names=["ML", "PIML 1", "PIML 2", "PIML 3"], hsize=12Gadfly.inch, logy=true)