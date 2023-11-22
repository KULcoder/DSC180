# DSC180A Project 1: Neural Network Feature Development

This project explores the question of how features evolve during the training process. For example, do they become more orthogonal? To investigate this, we will be examining the MNIST dataset: a large database consisting of handwritten digits ranging from 0 to 9, where each image is 28x28 pixels in size. We build a 5-layer ResNet-18 to train on the MNIST dataset, we will then investigate how features evolve during the training process (in each epoch) and possibly investigate how the CNFA or the AGOP changes in the process.

## Retrieving the data locally: 
- data can be downloaded directly via the PyTorch 'torchvision.datasets' class.

## Reference Paper
https://arxiv.org/abs/2309.00570 

## Agenda
1. Complete model save and load functionality
2. Set up forward hooker to record output and gradient of the first convolutional layer
3. Extract the fully connected layer's weight w (at the end of our model), then try visualize $w^T w$ 
4. Extract the first colutional layer's weight, try dot product matrix
5. Write a wrapped version in run.py: record everthing preperly by files
6. Write a proper dockerfile
7. Correct way to visualize output after each layer


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
  - data
  - models
    - build_model
    - train_model
    - evaluate_model
  - visualization
- README.md
- requirements.txt
- run.py

## Env
Use `conda env create -f environment.yml` to recreate the conda environment

### Notice
#### `.gitignore`
- **data folder** is ignored (don't upload MNIST, pytorch will help download it on your computer)
- **jupyter notebook checkpoint**
- **__pycache__**

