import Flux
import CUDA
import Mads
import Random
import Gadfly

function ml0_model(; input=3, output=100, device=Flux.gpu)
	model = device(Flux.Chain(Flux.Dense(input, 64), Flux.Dense(64, output)))
	return model, Flux.params(model)
end

function getdata(Xi, Xo, args)
    ir = sortperm(rand(size(Xi, 2)))
    c = Int(floor(size(Xi, 2) * 0.8))
	xtrain = Xi[:, ir[1:c]]
	ytrain = Xo[:, ir[1:c]]
	Mads.plotseries(ytrain, joinpath(workdir, "figures", "Additive Training Data.png"); title="Training Data", name="", xtitle="Time", ytitle="Value")
	xtest = Xi[:, ir[c+1 : end]]
	ytest = Xo[:, ir[c+1 : end]]
	Mads.plotseries(ytest, joinpath(workdir, "figures", "Additive Testing Data.png"); title="Testing Data", name="", xtitle="Time", ytitle="Value")

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
	epochs::Int = 4000       # number of epochs
	use_cuda::Bool = false   # use gpu (if cuda available)
end

function train(Xi, Xo, ml_model; kws...)
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
	model, params = ml_model(; input=size(Xi, 1), output=size(Xo,1), device=device)

	train_loader, test_loader = getdata(Xi, Xo, args)

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
		if epoch % 1000 == 0
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

model0, train_loss0, test_loss0 = train(C_statistics_n, C_transient_mass_n[1:100:1000,:], ml0_model)

Mads.plotseries([C_transient_mass_n[1:100:1000,150] model0(C_statistics_n[:,150])])

model0bm, train_loss0bm, test_loss0bm = train([bm'; C_statistics_n], C_transient_mass_n[1:100:1000,:], ml0_model)

Mads.plotseries([C_transient_mass_n[1:100:1000,150] model0bm([bm[150,:]; C_statistics_n[:,150]])])