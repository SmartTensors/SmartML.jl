module SmartML

import Mads
import NMFk
import MLJ
import SVR
import Printf
import Suppressor

if Base.source_path() !== nothing && Base.source_path() != ""
	const smartmldir = first(splitdir(first(splitdir(Base.source_path()))))
else
	const smartmldir = "."
end

workingdir = "."

function setworkingdir(dirname::AbstractString)
	global workingdir = dirname
end

workdir = "."

include("SmartML_Model.jl")
include("SmartML_Mads.jl")

end