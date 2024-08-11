module SmartML

import MLJ
import MLJXGBoostInterface
import MLJLIBSVMInterface
import XGBoost
import Mads
import NMFk
import SVR
import Printf
import Suppressor
import DelimitedFiles
import Gadfly
import JLD
import Statistics
import Interpolations

smlmodel = Union{SVR.svmmodel, MLJ.Machine}

const smartmldir = Base.pkgdir(SmartML)
workdir = smartmldir

function setworkdir(dirname::AbstractString)
	global workdir = dirname
end

include("SmartML_Data.jl")
include("SmartML_Models.jl")
include("SmartML_Model.jl")
include("SmartML_Mads.jl")
include("SmartML_Analysis.jl")

end