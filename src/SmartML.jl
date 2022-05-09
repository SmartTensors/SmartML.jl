module SmartML

import SVR
import NMFk
import Mads
import Printf
import Suppressor

if Base.source_path() !== nothing
	const dir = first(splitdir(first(splitdir(Base.source_path()))))
end

include("SmartML_Model.jl")
include("SmartML_Mads.jl")

end