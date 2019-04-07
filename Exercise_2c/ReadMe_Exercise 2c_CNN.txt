To complete model_task2c.py, a CNN implementation, following parameters were calculated from the given script:

  * from the MNIST image size: self.expected_input_size = (28, 28)

  * 3 channels from MNIST images (though they are b/w only)

  * The in_features=1536 are the product of out_channels and the dimension of the output image from nn.Conv2d:
    - 1536 = 3 * 512 =  3 x 2^9, thus the output image from nn.Conv2d should be something with 2^k
    - Now, 28 = 3 x 8 + 4 so the kernel_size is 7 and the output image has dimensions 8 x 8

  * for the 10 classes, the 10 digits: nn.Linear(1536, 10)
