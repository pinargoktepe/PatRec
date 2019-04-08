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


In general, accuracy to classify train data increased with increasing epochs and plateaued starting at epoch = 30 over 95 % accuracy. It was higher at a higher learning rate of 0.1.

![Accuracy to classify train data](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/train.png)


Accuracy to predict the test data showed similar with best results at a learning rate of 0.1 with accuracy of > 98 % at all tested values of epochs. But also a learning rate of 0.05 gave almost the same results.
![Accuracy to predict test data](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/test.png)

Combined from this results, a learning rate of 0.1 and 30 epochs were further used.

# Effect of random initialization
For this, the flag `<--mulit-run 5>` were set what runs the CNN with 5 different random initializations.

The outcome accuracy changed slightly (98.242, 98.425, 98.275, ...) ...


# Accuracy with the best parameters found
![Confusion matrix validation](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/CNN_30_0.1_val_confusion_matrix_29.png)

![Confusion matrix test](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/CNN_30_0.1_test_confusion_matrix_29.png)
