## Three-Dimensional Permeability Inversion Using Convolutional Neural Networks and Positron Emission Tomography
[![DOI](https://zenodo.org/badge/265937462.svg)](https://zenodo.org/badge/latestdoi/265937462)

The first step in understanding how water and contaminants are flowing in the subsurface is to describe the ease at which fluid can flow—a hydrogeologic property termed permeability. Variation in permeability is an intrinsic property of geologic materials that arises due to differences in the underlying geologic processes that generated the materials. Recent utilization of in situ imaging, specifically positron emission tomography (PET), enables the measurement of three-dimensional (3-D) time-lapse radiotracer solute transport in geologic porous media. 

Leveraging the relationship between local permeability variation and solute advection behavior, an encoder-decoder based convolutional neural network (CNN) is implemented as a permeability inversion scheme using a single PET scan of a radiotracer pulse injection experiment as input. Compared to traditional mathematical modeling approaches, the trained deep learning model significantly reduces the computational cost while accurately predicting the 3-D permeability distributions in real geologic materials. Details related to this work can be found in <a href="https://doi.org/10.1029/2021WR031554">here</a>.

An example of the inversion output for a Berea sandstone core is illustrated below.
<p align="center">
<img src="./CNN_final_version/Figures/f7.jpg" width="800px"></img>
</p>

### Workflow
The figure below is a schematic illustration of the inversion-validation workflow using both synthetic (top loop) and experimental PET data (bottom loop). The purple blocks correspond to synthetic/predicted permeability maps, the red block is the PET data, the orange blocks are experimental and modeled arrival time difference maps. The CNN components include convolutional blocks (blue), up/down-sampling block (yellow), and residual-in-residual dense blocks (green).


The network training was a supervised process. The inputs were 3-D image tensors containing the arrival time difference maps (<img src="./CNN_final_version/Figures/y.gif"></img>) and the permeability maps of the corresponding synthetic geologic core were the target data (<img src="./CNN_final_version/Figures/x.gif"></img>). The predicted permeability maps (<img src="./CNN_final_version/Figures/x_hat.gif"></img>) were compared with the target synthetic permeability maps through loss functions. Training accuracy was evaluated on the test set by individually comparing each synthetic permeability map with the network predicted permeability map or by comparing the experimental arrival time difference maps and the modeled arrival time difference maps (<img src="./CNN_final_version/Figures/yhat.gif"></img>) based on the network permeability map prediction.
<p align="center">
<img src="./CNN_final_version/Figures/workflow_figv2.jpg" width="800px"></img>
</p>

### Example
The following is a <a href="https://github.com/zahasky/Neural_network_inversion/blob/master/CNN_final_version/jupyter_workflow/LoopValidation.ipynb">jupyter notebook</a> demonstrating the overall workflow. Since the jupyter notebook reads files a bit differently than the normal IDE, please use the file in the jupyter_workflow folder. 

### Citation
If you use any of the associated data, codes, or models, the paper that desribes this work can be cited using the following bibtex entry.
```
@article{article,
author = {Huang, Zitong and Kurotori, Takeshi and Pini, Ronny and Benson, S. and Zahasky, Christopher},
year = {2022},
month = {03},
pages = {},
title = {Three‐Dimensional Permeability Inversion Using Convolutional Neural Networks and Positron Emission Tomography},
journal = {Water Resources Research},
doi = {10.1029/2021WR031554}
}
```
