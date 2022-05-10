SmartML
-----

**SmartML** is one of the tools in the **SmartTensors** ML framework ([smarttensors.com](https://smarttensors.com)).

<div style="text-align: left">
    <img src="logo/SmartTensorsNewSmall.png" alt="SmartTensors" width=25%  max-width=125px;/>
</div>

**SmartML** applied unsupervised and supervised machine learning methodology that allows for automatic identification of the optimal number of features (signals/signatures) present in the data.


**SmartML** can be applied to perform:
- Feature extraction (**FE**)
- Blind source separation (**BSS**)
- Detection of disruptions / anomalies
- Image recognition
- Text mining
- Data classification
- Separation (deconstruction) of co-occurring (physics) processes
- Discovery of unknown dependencies and phenomena
- Development of reduced-order/surrogate models
- Identification of dependencies between model inputs and outputs
- Guiding the development of physics models representing the ML analyzed data
- Blind predictions
- Optimization of data acquisition (optimal experimental design)
- Labeling of datasets for supervised ML analyses

**SmartML** provides high-performance computing capabilities to solve problems with Shared and Distributed Arrays in parallel.
The parallelization allows for utilization of multi-core / multi-processor environments.
GPU and TPU accelerations are available through existing Julia packages.

**SmartML** provides advanced tools for data visualization, pre- and post-processing.
These tools substantially facilitate utilization of the package in various real-world applications.

**SmartML** methodology and applications are discussed in the research papers and presentations listed below.

**SmartML** is demonstrated with a series of examples and test problems provided here.

## Awards

**SmartTensors** and **SmartML** were recently awarded:
* 2021 R&D100 Award: [Information Technologies (IT)](https://www.rdworldonline.com/2021-rd-100-award-winners-announced-in-analytical-test-and-it-electrical-categories)
* 2021 R&D100 Bronze Medal: [Market Disruptor in Services](https://www.rdworldonline.com/2021-rd-100-special-recognition-winners-announced)

<div style="text-align: left">
    <img src="logo/RD100Awards-300x300.png" alt="R&D100" width=25%  max-width=125px;/>
</div>

## Installation

After starting Julia, execute:

```julia
import Pkg
Pkg.add("SmartML")
```

to access the latest released version.

To utilize the latest code updates (commits), use:

```julia
import Pkg
Pkg.add(Pkg.PackageSpec(name="SmartML", rev="master"))
```