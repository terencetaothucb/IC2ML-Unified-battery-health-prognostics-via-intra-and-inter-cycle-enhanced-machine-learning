# IC2ML: Unified battery state-of-health, degradation trajectory and remaining useful life prediction via intra-nd inter-cycle enhanced machine learning 
Strategic management of lithium-ion batteries (LIBs) depends on evaluating current health status and predicting future degradation paths. Yet despite extensive research on core management tasks like 
state of health (SOH) estimation, degradation trajectory prediction, and remaining useful life (RUL) prediction, these tasks remain isolated without leveraging their inherent connections. This work 
proposes an unified framework that enables joint battery SOH, degradation trajectory and RUL prediction via an intra-cycle and inter-cycle enhanced machine learning (IC2ML). The IC2ML uses 1-D time-serials 
voltage data to implement SOH prediction, where the inter-cycle embeddings are further self-attentioned for degradation trajectory prediction. The RUL is derived from degradation trajectory prediction based 
on anticipated SOH levels, enabled by cross attention between output embeddings and input inter-intra cycle embeddings. The results demonstrate that using only 0.1V sampling interval data that can be extracted 
on-site, the average average root mean square error for SOH, degradation trajectory, and RUL prediction is 1.85%, 2.36% and 23.90 cycles, respectively, validated experimentally on 121 batteries spanning 
10 operation conditions. Sensitivity analysis shows that IC2ML can be adapted to scenarios where a few historical data is accessible. Broadly, this work highlights the significant poteintial of strategical 
battery management algorithm co-design using intra-cycle and inter-cycle battery degradation information for various management tasks.

## Highlights
- IC2ML, a unified framework jointly predicting SOH, degradation trajectory, and RUL, is proposed. 
- Health indicator are extracted from both 1-D voltage time series and 2-D images of voltage-capacity data.
- Spatiotemporal interaction among SOH, degradation trajectory and RUL is implemented through attention-based methods.
- The generalizability of IC2ML is validated with batteries of 3 materials and 10 operating conditions.
- IC2ML can adapt to limited data and extend to 100-cycle trajectory prediction with 1.77% RMSE.

# 1. Setup
## 1.1 Enviroments
* Python (Jupyter notebook) 
## 1.2 Python requirements
* python=3.11.5
* numpy=1.26.4
* torch=2.4.1
* keras=2.15.0
* matplotlib=3.9.2
* scipy=1.13.1
* scikit-learn=1.3.1
* pandas=2.2.2

# 2. Datasets
The raw data can be accessed via the following link:
* [Dataset](https://doi.org/10.5281/zenodo.6379165)

# 3. Demo
We provide a detailed demo of our code running .
1. Run the `run.py` file to train our model. The program will generate a folder named `checkpoint` and save the results in it.
2. You can change `setattr(args,'dataset')` to select the NCA, NCM, NCANCM datasets. It will generate a folder in the `checkpoint` to save the results of the corresponding datasets.

**Note: The results presented in this paper were not obtained through specific hyperparameter optimization. You may experiment with alternative hyperparameters to achieve similar or potentially improved outcomes.
Due to the inherent stochasticity of neural networks, the acquired expert weights will not remain identical across different runs. However, it is evident that significant differences exist among distinct aging stages.**
## Acknowledgement
This repo is constructed based on the following repos:
- https://github.com/thuml/Time-Series-Library
- Thanks to the following code for the assistance it provided in this paper.
- https://github.com/terencetaothucb/Early-Battery-Degradation-Prediction-via-Chemical-Process-Inference
- https://zenodo.org/records/15350607
- https://github.com/wang-fujin/PINN4SOH
