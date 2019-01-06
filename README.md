# Class-Imbalance-in-WW-Polarization
Treating the measurement of the same-sign W polarization fraction as a class imbalance problem

## Motivation
A couple of papers, arXiv:1510.01691, arXiv:1812.07591, have used deep learning to determine the polarization fraction, _W<sub>L</sub> W<sub>L</sub> / &Sigma;<sub>i, j</sub> W<sub>i</sub> W<sub>j</sub>_, in same-sign _WW_ scattering. 

In this reaction two protons (_p_) collide at the Large Hadron Collider and produce two jets (_j_), collimated sprays of hadronic particles, and two _W_ bosons with the same electric charge. This process is interesting as a probe of the unitarization (probability conservation) mechanism in the Standard Model (SM) of particle physics.

The polarization fraction is predicted to be small in the SM, ~5%. Thus there is an imbalance of events where both _W_'s are longitudinally polarized vs. when one or none is longitudinally polarized. This motivates trying to treat this as a class imbalance problem, something which neither of the above papers do.

## Setup
Clone repository
```
git clone https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization
```

## Requisites

- Anaconda Python 3
- Imbalanced Learn `conda install -c conda-forge imbalanced-learn`
- Keras version 2.2 or higher `conda install -c conda-forge keras`
Imbalanced Learn is not fully compatible with Keras v2.0. I am not sure about v2.1.
- pandas, NumPy, scikit-learn, TensorFlow, Matplotlib, SciPy, and Jupyter Notebook
Optionally:
- pylhe `pip install pylhe` This is only used in preprocessing to extract the info in lhe files and write them to csv. Since the csv files are included in the data folder there is no need to install this package to use the rest of the repo.

## Analysis
See the [notebook](https://github.com/christopher-w-murphy/Class-Imbalance-in-WW-Polarization/blob/master/comparison.ipynb) for full details.

First is a comparison between this work and arXiv:1812.07591, which establishes some baseline models

Next these models are evaluated using metrics better suited for class imbalance problems.

Then difference model, aimed specifically aim class inbalance, are trained to see if there is an improvement in performance. These include:
- Class Weights
- Balanced Random Forest
- Focal Loss
- Balanced Batch Generator
