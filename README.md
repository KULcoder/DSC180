# DSC180A Project 1: Neural Network Feature Development

This project explores the question of how features evolve during the training process. For example, do they become more orthogonal? To investigate this, we will be examining the MNIST dataset: a large database consisting of handwritten digits ranging from 0 to 9, where each image is 28x28 pixels in size. We build a 5-layer ResNet-18 to train on the MNIST dataset, we will then investigate how features evolve during the training process (in each epoch) and possibly investigate how the CNFA or the AGOP changes in the process.

## Example usage
1. Set up environment by `conda env create -f environment.yml`.
2. Check `notebooks/0.0-Example-Usage.ipynb` file: it tells how the source code works.
3. Check `notebooks/9.0-Q1-Result.ipynb` file: it includes the code how we obatin our result.

### Retrieving the data locally: 
- data downloaded directly via the PyTorch 'torchvision.datasets' class.

## Reference Paper
https://arxiv.org/abs/2309.00570 

## Agenda
1. Set up hooks to record layer outputs and gradients of the first convolutional layer
2. Extract the first colutional layer's weight, try dot product matrix
3. Find methods to implement AGOP and compare it with $w^Tw$
4. Write a wrapped version in run.py: record everthing preperly by files
5. Write a proper dockerfile
6. Correct way to visualize output after each layer


## Structure
- data
  - raw data
- config
  - example_config.json
  - configs.md
- notebooks
- reports
  - figures
- src
  - \_\_init\_\_.py
  - experiment
    - Experiment
    - utils
  - data
    - get_dataloader
  - models
    - get_model
    - init_weights
    - model_files
    - model
  - visualization
    - acc_loss
- README.md
- requirements.txt
- run.py (Not Implemented)

## Env
Use `conda env create -f environment.yml` to recreate the conda environment

### Notice
#### `.gitignore`
- **data folder** is ignored (don't upload MNIST, pytorch will help download it on your computer)
- **jupyter notebook checkpoint**
- **__pycache__**

#### Notebook Naming Rules
1. Numbers before:
Any file started with `0.n` is example usage notebook for reference usage.

**Suggestions**: 
- Use `1.n` for training model
- Use `2.n` for visualizing model
- Use `3.n` for analyzing model
- Use `9.n` for experiment results

