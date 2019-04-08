# Complete CNN in model_task2c.py
To complete model_task2c.py, a CNN implementation, following parameters were calculated from the given script:

  * from the MNIST image size: self.expected_input_size = (28, 28)
  * 3 channels from MNIST images (though they are b/w only)
  * The in_features=1536 are the product of out_channels and the dimension of the output image from nn.Conv2d:
    - 1536 = 3 * 512 =  3 x 2^9, thus the output image from nn.Conv2d should be something with 2^k
    - Now, 28 = 3 x 8 + 4 so the kernel_size is 7 and the output image has dimensions 8 x 8
  * for the 10 classes, the 10 digits: nn.Linear(1536, 10)

# Manually optimize accuracy depending on parameters
To optimize the accuracy of prediction, the values of learning rate and number of epochs were iteratively modified (cells with 'x' were actually run):

| Epochs/Learning rate | 0.001 | 0.005 | 0.01 | 0.05 | 0.1 |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 10 | x | x | x | x | x |
| 20 | x |   |   |   |   |
| 30 | x | x | x | x | x |
| 40 | x |   |   |   |   |
| 50 | x | x | x | x | x |


**Temporary graphs:**
Accuracy to classify train data...
![Accuracy to classify train data](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/train.png)


Accuracy to predict test data...
![Accuracy to predict test data](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/test.png)


# Effect of random initialization
Results coming soon...

# Accuracy with the best parameters found
Results coming soon...
