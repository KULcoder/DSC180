# DSC180A Project 1: Neural Network Feature Development

This project explores the question of how features evolve during the training process. For example, do they become more orthogonal? To investigate this, we will be examining the MNIST dataset: a large database consisting of handwritten digits ranging from 0 to 9, where each image is 28x28 pixels in size. We build a 5-layer ResNet-18 to train on the MNIST dataset, we will then investigate how features evolve during the training process (in each epoch) and possibly investigate how the CNFA or the AGOP changes in the process.

## Retrieving the data locally: 
- data can be downloaded directly via the PyTorch 'torchvision.datasets' class.

## Agenda
1. Write a wrapped version in run.py: record everything properly by files
2. Write requirements.txt
3. Complete checkpoint and load-in functionality (consider write a function to load config?)

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

### Notice
#### `.gitignore`
- **data folder** is ignored (don't upload MNIST, pytorch will help download it on your computer)
- **jupyter notebook checkpoint**
- **__pycache__**
