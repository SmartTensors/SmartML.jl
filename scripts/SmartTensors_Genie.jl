import Stipple
import Stipple: @with_kw, @reactors, R, ChannelName
import Genie
import StippleUI
import StipplePlotly
import CSV
import JLD
import JLD2
import FileIO
import DataFrames
import NMFk
import Mads
import Gadfly
import Colors
import ColorSchemes
import Clustering

Genie.config.cors_headers["Access-Control-Allow-Origin"]  =  "*"
Genie.config.cors_headers["Access-Control-Allow-Headers"] = "Content-Type"
Genie.config.cors_headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
Genie.config.cors_allowed_origins = ["*"]

datasetdir = joinpath(@__DIR__, "..", "gui-data")
figuredir = joinpath(@__DIR__, "..", "gui-results")
Mads.mkdir(datasetdir)
Mads.mkdir(figuredir)

@Stipple.reactive mutable struct DataModel <: Stipple.ReactiveModel
	rerun::Stipple.R{Int} = 0
	datasets_reload::Stipple.R{Int} = 0
	dataset::Stipple.R{String} = ""
	datasets::Stipple.R{Vector{String}} = readdir(datasetdir)
	datatable::Stipple.R{StippleUI.DataTable} = StippleUI.DataTable()
	datatable_pagination::StippleUI.DataTablePagination = StippleUI.DataTablePagination(rows_per_page=200)

	method::Stipple.R{String} = ""
	methods::Stipple.R{Vector{String}} = ["k-means", "NMFk"]
	attributes::Stipple.R{Vector{String}} = []
	x_attribute::Stipple.R{String} = ""
	y_attribute::Stipple.R{String} = ""
	z_attribute::Stipple.R{String} = ""

	data_plot::Stipple.R{Vector{StipplePlotly.PlotData}} = []
	cluster_plot::Stipple.R{Vector{StipplePlotly.PlotData}} = []
	layout::Stipple.R{StipplePlotly.PlotLayout} = StipplePlotly.PlotLayout(plot_bgcolor = "#fff")

	no_of_clusters::Stipple.R{Int} = 3
	no_of_iterations::Stipple.R{Int} = 10
end

function load_datasets!(smarttensors_model::DataModel, datadir=datasetdir)
	files = readdir(datadir)
	smarttensors_model.datasets[] = files
	return nothing
end

function load_data!(smarttensors_model::DataModel, filename)
	if typeof(filename) <: AbstractString
		f = filename
	else
		f = filename[]
	end
	e = lowercase(last(splitext(f)))
	if  e == ".csv"
		data_input = CSV.read(joinpath(datasetdir, f), DataFrames.DataFrame)
	elseif e == ".jld2"
		data_input = FileIO.load(joinpath(datasetdir, f), "df")
	elseif e == ".jld"
		data_input = JLD.load(joinpath(datasetdir, f), "df")
	end
	smarttensors_model.attributes[] = names(data_input)
	data_input.Cluster = repeat(["-"], size(data_input, 1))
	smarttensors_model.datatable[] = StippleUI.DataTable(data_input)
	return nothing
end

function plot_data(smarttensors_model::DataModel, cluster_column::Symbol)
	plot_collection = Vector{StipplePlotly.PlotData}()
	(isempty(smarttensors_model.x_attribute[]) || isempty(smarttensors_model.y_attribute[])) && return plot_collection

	@info "Plotting data $(smarttensors_model.x_attribute[]), $(smarttensors_model.y_attribute[]), $(cluster_column)..."
	df = smarttensors_model.datatable[].data
	m = Matrix(df)
	use_columns = vec(sum(typeof.(m) .<: AbstractString; dims=1)) .== 0
	n = smarttensors_model.attributes[][use_columns[1:end-1]]
	@info("Data attributes: $(n)")
	c = first(indexin([string(cluster_column)], smarttensors_model.attributes[]))
	if c !== nothing || cluster_column == :Cluster
		@info("Plotting data for $(cluster_column) ...")
	else
		@warn("Cannot plot data for $(cluster_column)!")
		return plot_collection
	end
	if cluster_column == :Cluster || use_columns[c] == false
		for species in Array(df[:, cluster_column]) |> unique!
			if species == "-"
				continue
			end
			x_attribute_collection, y_attribute_collection = Vector{Float64}(), Vector{Float64}()
			for r in eachrow(df[df[!, cluster_column] .== species, :])
				push!(x_attribute_collection, (r[Symbol(smarttensors_model.x_attribute[])]))
				push!(y_attribute_collection, (r[Symbol(smarttensors_model.y_attribute[])]))
			end
			plot = StipplePlotly.PlotData(
						x = x_attribute_collection,
						y = y_attribute_collection,					
						mode = "markers",
						name = string(species),
						plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER)
			push!(plot_collection, plot)
		end
	else		
		z = df[!, Symbol(smarttensors_model.z_attribute[])]
		plot = StipplePlotly.PlotData(
					x = df[!, Symbol(smarttensors_model.x_attribute[])],
					y = df[!, Symbol(smarttensors_model.y_attribute[])],
					mode = "markers",
					marker = StipplePlotly.PlotDataMarker(color=z, colorscale="Viridis", size=14, colorbar=StipplePlotly.ColorBar(thickness=20)),
					plot = StipplePlotly.Charts.PLOT_TYPE_SCATTER)
		push!(plot_collection, plot)
	end
	smarttensors_model.layout[] = StipplePlotly.PlotLayout(plot_bgcolor = "#fff",
	xaxis = [StipplePlotly.PlotLayoutAxis(xy = "x", index = 1, title = smarttensors_model.x_attribute[], ticks="outside", showline = false, zeroline = true)],
	yaxis = [StipplePlotly.PlotLayoutAxis(xy = "y", index = 1, title = smarttensors_model.y_attribute[], ticks="outside", showline = false, zeroline = true)])	
	return plot_collection
end

function compute_clusters!(smarttensors_model::DataModel)
	p =  first(splitext(last(splitdir(smarttensors_model.dataset[]))))
	df = smarttensors_model.datatable[].data
	use_columns = vec(sum(typeof.(Matrix(df)) .<: AbstractString; dims=1)) .== 0
	# display(use_columns)
	# display(smarttensors_model.attributes[])
	data_columns = smarttensors_model.attributes[][use_columns[1:end-1]]
	m = smarttensors_model.method[]
	@info("Processing data attributes: $(data_columns) using method $(m)... ")
	data = collect(Matrix(df[:, [Symbol(c) for c in data_columns]]))
	label_columns = smarttensors_model.attributes[][.!use_columns[1:end-1]]
	if length(label_columns) > 0
		@info("Label attributes: $(label_columns)")
		labels = string.(vec(collect(Matrix(df[:, [Symbol(c) for c in label_columns]]))))
	else
		labels = ["w$i" for i = 1:size(W, 1)]
	end
	@show labels
	im = ismissing.(data)
	data[im] .= NaN
	data = float.(data)
	if m == "k-means"
		@info("Computing k-means clusters...")
		result = Clustering.kmeans(permutedims(data), smarttensors_model.no_of_clusters[]; maxiter=smarttensors_model.no_of_iterations[])
		ca = Clustering.assignments(result)
		l = Vector{String}(undef, size(ca, 1))
		for i in sort(unique(ca))
			c = '@' + i
			l[ca .== i] .= "$(c)"
		end
	elseif m == "NMFk"
		W, H, o, s, a = NMFk.execute(first(NMFk.normalizematrix_col(data)), smarttensors_model.no_of_clusters[], smarttensors_model.no_of_iterations[]; load=true, resultdir=joinpath(figuredir, p), casefilename=p)
		o, Wl, Hl = NMFk.postprocess(W, H, labels, data_columns; Wcasefilename="locations", Hcasefilename="attributes", resultdir=joinpath(figuredir, p), figuredir=joinpath(figuredir, p))
		l = first(Wl)
	else
		@warn("Unknown method $(m)!")
		return nothing
	end
	df[!, :Cluster] = l
	smarttensors_model.datatable[] = StippleUI.DataTable(df)
	smarttensors_model.cluster_plot[] = plot_data(smarttensors_model, :Cluster)
	@info("Cluster plot done!")
	@info("Processing done!")
	return nothing
end

function ui_smarttensors(smarttensors_model::DataModel)
	Stipple.on(smarttensors_model.datasets_reload) do (_...)
		@info "Reload ..."
		load_datasets!(smarttensors_model)
	end
	Stipple.on(smarttensors_model.rerun) do (_...)
		@info "Rerun ..."
		compute_clusters!(smarttensors_model)
	end
	Stipple.onany(smarttensors_model.dataset) do (_...)
		load_data!(smarttensors_model, smarttensors_model.dataset[])
	end
	Stipple.onany(smarttensors_model.method, smarttensors_model.no_of_clusters, smarttensors_model.no_of_iterations) do (_...)
		load_datasets!(smarttensors_model)
		if !isempty(smarttensors_model.z_attribute[])
			smarttensors_model.data_plot[] = plot_data(smarttensors_model, Symbol(smarttensors_model.z_attribute[]))
			@info("Data plot done!")
		end
		compute_clusters!(smarttensors_model)
		@info("Computation done!")
	end
	Stipple.onany(smarttensors_model.x_attribute, smarttensors_model.y_attribute, smarttensors_model.z_attribute) do (_...)
		load_datasets!(smarttensors_model)
		if !isempty(smarttensors_model.z_attribute[])
			smarttensors_model.data_plot[] = plot_data(smarttensors_model, Symbol(smarttensors_model.z_attribute[]))
			@info("Data plot done!")
		end
		smarttensors_model.cluster_plot[] = plot_data(smarttensors_model, :Cluster)
		@info("Cluster plot done!")
	end
	Stipple.page(smarttensors_model, class="container", title="SmartTensors", head_content="<link rel=\"icon\" type=\"image/x-icon\" href=\"\"/>",
		prepend=Stipple.style("""
			tr:nth-child(even) {
				background: #F8F8F8 !important;
			}
			.modebar {
				display: none!important;
			}
			.st-module {
				background-color: #FFF;
				border-radius: 2px;
				box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.04);
			}
			.stipple-core .st-module > h5
			.stipple-core .st-module > h6 {
				border-bottom: 0px !important;
			}"""),
		[
			StippleUI.heading("Carbon Solutions LLC"; img=Stipple.img(src="https://www.carbonsolutionsllc.com/wp-content/uploads/2021/07/cropped-CS-logo_LargeRectangular_TransparentBackground.png", style="width:80px; height:100px; vertical-align:middle; margin-right:10px;"))
			Stipple.row([
				Stipple.cell(class="st-module", [
					Stipple.img(src="http://smarttensors.com/logo/SmartTensorsNew.png", style="width:30px;height:30px;vertical-align:middle;margin-right:10px;")
					Stipple.h4("SmartTensors: Smart Data Mining")
					])
			])
			Stipple.row([
				Stipple.cell(class="st-module", [StippleUI.uploader(label="Upload Dataset:", :auto__upload, method="POST", url="http://localhost:9000/", field__name="csv_file")
				])
				Stipple.cell(class="st-module", [
					Stipple.button("Reload the data directory!", @StippleUI.click("datasets_reload += 1"))
				])
				Stipple.cell(class="st-module", [
					Stipple.button("Rerun!", @StippleUI.click("rerun += 1"))
				])
			])
			Stipple.row([
				Stipple.cell(class="st-module", [
					Stipple.h6("Dataset:")
					Stipple.select(:dataset; options=:datasets)
				])
				Stipple.cell(class="st-module", [
					Stipple.h6("Method:")
					Stipple.select(:method; options=:methods)
				])
			])
			Stipple.row([
				Stipple.cell(class="st-module", [
					Stipple.h5("Data:")
					Stipple.table(:datatable; pagination=:datatable_pagination, dense=true, flat=true, style="height: 350px;")
				])
			])
			Stipple.row([
				Stipple.cell(class="st-module", [
					Stipple.h6("Number of clusters:")
					StippleUI.slider(1:1:20, @Stipple.data(:no_of_clusters); label=true)
				])
				Stipple.cell(class="st-module", [
					Stipple.h6("Number of iterations:")
					StippleUI.slider(10:10:200, @Stipple.data(:no_of_iterations); label=true)
				])
				Stipple.cell(class="st-module", [
					Stipple.h6("X attribute:")
					Stipple.select(:x_attribute; options=:attributes)
				])
				Stipple.cell(class="st-module", [
					Stipple.h6("Y attribute:")
					Stipple.select(:y_attribute; options=:attributes)
				])
				Stipple.cell(class="st-module", [
					Stipple.h6("Z attribute:")
					Stipple.select(:z_attribute; options=:attributes)
				])
			])
			Stipple.row([
				Stipple.cell(class="st-module", [
					Stipple.h5("Data:")
					StipplePlotly.plot(:data_plot; layout=:layout, config="{displayLogo:false}")
				])
				Stipple.cell(class="st-module", [
					Stipple.h5("Extracted clusters:")
					StipplePlotly.plot(:cluster_plot; layout=:layout, config="{displayLogo:false}")
				])
			])
			# Stipple.row([
			# 	Stipple.cell(class="st-module", [
			# 		Stipple.img(src="http://smarttensors.com/logo/SmartTensorsNew.png", style="width:30px;height:30px;vertical-align:middle;margin-right:10px;")
			# 	])
			# ])
		]
	)
end

Stipple.route("/") do
	m = Stipple.init(DataModel)
	Stipple.html(ui_smarttensors(m))
end

#uploading csv files to the backend server
Stipple.route("/", method=Stipple.POST) do
	files = Genie.Requests.filespayload()
	for f in files
		d = joinpath(datasetdir, f[2].name)
		write(d, f[2].data)
		@info "Uploading: " * f[2].name * "(destination: " * d * ")"
		return f[2].name
	end
	if length(files) == 0
		@warn "No file uploaded"
		return nothing
	end
end

Stipple.up(9000; async=true, server=Stipple.bootstrap())
Stipple.down()