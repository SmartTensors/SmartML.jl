module SmartML

import Mads
import NMFk
import MLJ
import SVR
import Printf
import Suppressor

const smartmldir = Base.pkgdir(SmartML)
workdir = smartmldir

function setworkdir(dirname::AbstractString)
	global workdir = dirname
end

include("SmartML_Model.jl")
include("SmartML_Mads.jl")

end