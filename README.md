# DA6401_Assignment_2


## Notes
### Dropout placement
`Conv2D → BatchNorm → ReLU → Dropout → Pooling` 

### Batch Normalization (BN) Placement
After Convolution, Before Activation: BN is typically placed after the convolutional layer but before the activation function (e.g., ReLU) to normalize inputs for non-linearities