# New Initialization Mechanisms for Convolutional Neural Networks

This project explores new ways of initialization convolutional neural networks utilizing Average Gradient Outer Product (AGOP) and Neural Feature Matrix (NFM). 
Previous study suggests that AGOP and NFM characterized the feature learning process of deep learning models, and we are curious about what if we utilize those 
information to initialize the network. Afterall, why models are always started from random?

## How to Run
1. Set up environment by `conda env create -f environment.yml`. (or alternative methods below)
2. In the root directory, run `python3 src/model/save_nfm_agop.py <config_file>` to save the neural feature matrix and averge gradient outer product to local directory for further training.
   - The saved matrices for each layer will be saved under the folder specified in the config file.
3. In the root directory, run `python3 experiments/train_test.py <config_file>` to train a model according to that config file.
   - result will be saved into `logs` folder
   - if `save_model=true`, model will be saved into `models` folder

### Retrieving the data locally: 
- Dataset `CIFAR10`, `CIFAR100`, `SVHN` are downloaded directly via the PyTorch 'torchvision.datasets' class.
- Dataset `Tiny-ImageNet` is downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip

## Reference Paper

Mechanism of feature learning in convolutional neural networks
https://arxiv.org/abs/2309.00570 

Mechanism of feature learning in deep fully connected networks and kernel machines that recursively learn features
https://arxiv.org/abs/2212.13881

Mechanism for feature learning in neural networks and backpropagation-free machine learning models
https://www.science.org/doi/10.1126/science.adi5639


## Env
Use `conda env create -f environment.yml` to recreate the conda environment

### Alternative Methods

conda create -n env_name python=3.9

conda activate env_name

conda config --env --add channels conda-forge
conda install numpy

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install tqdm

conda install scipy

conda install matplotlib


