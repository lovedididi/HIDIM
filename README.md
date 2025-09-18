# HIDIM
## A Heterogeneous Integration and Decoupled Identification Model for Imbalanced Identification of Wheat Unsound Kernels
## Authors: Panpan Wanga, Yuan Yao, Peng Guo, Lei Li, Heling Cao

## Usage
Due to the large size of the dataset, it is not stored in this file. If necessary, please redirect to [GrainSpace]( https://grainnet.github.io/GrainSpace.html).  
Run run. py. After training and evaluation, obtain the final indicators.  
If you have already obtained the parameters of the first part, you can choose to run the second part directly in Classification.py.

### Package
We train and evaluate our HIDIM using an NVIDIA GeForce RTX 4090 GPU with 24 GB memory. Our code is based on PyTorch, and requires the following python packages:
- Python 3.8+
- PyTorch 2.4.1+cu118
- torchvision 0.19.1+cu118
- torchaudio 2.4.1+cu118
- timm 1.0.15
- numpy 1.24.1
- scikit-learn 1.3.2

