## Three-Dimensional Permeability Inversion Using Convolutional Neural Networks and Positron Emission Tomography
<p align="center">
<img src="./Figures/workflow_figv2.jpg" width="800px"></img>
</p>

The first step in understanding how water and contaminants are flowing in the subsurface is to describe the ease at which fluid can flow—a hydrogeologic property termed permeability. Variation in permeability is an intrinsic property of geologic materials that arises due to differences in the underlying geologic processes that generated the materials. Recent utilization of in situ imaging, specifically positron emission tomography (PET), enables the measurement of three-dimensional (3-D) time-lapse radiotracer solute transport in geologic porous media. 

Leveraging the relationship between local permeability variation and solute advection behavior, an encoder-decoder based convolutional neural network (CNN) is implemented as a permeability inversion scheme using a single PET scan of a radiotracer pulse injection experiment as input. Compared to traditional mathematical modeling approaches, the trained deep learning model significantly reduces the computational cost while accurately predicting the 3-D permeability distributions in real geologic materials. Details related to this work can be found in <a href="https://doi.org/10.1029/2021WR031554">here</a>.
