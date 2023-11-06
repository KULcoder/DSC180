# configs

- **experiment_name**: unique identifier for this run
- **visualize**: provide figure (not implemented)

## data
- **path**: path to file. 
    - Will be downloaded if not present
- **dataset**
    - choice: <u>"mnist"</u>
- **image_channels**
    - 1 for gray-scale (like MNIST), 3 for rgb (like CIFAR-10/ImageNet)
- **num_classes**
- **batch_size**
- **num_workers**: number of cpu for dataloader
- **val_split**: portion to split for validation set

## model
- **type**
    - choice: <u>"resnet18"</u>
- **first_layer_kernel_size**
    - first convolutional layer kernel size

## optimizer
- **type**
    - choice: <u>"sgd"</u>, <u>"adam"</u>
- **lr** (learning rate)
- **momentum**
- **weight_decay** (l2 regularization)
- **nestrov** (sepcfic for sgd)

## training
- **epochs**
- **criterion**: loss to use
    - choice: <u>"cross_entropy"</u>, <u>"mse"</u> 

