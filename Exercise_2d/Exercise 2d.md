# CNN (task 2c) on permutated MNIST dataset
Under the same parameters (epochs = 30, learning rate = 0.1), the two MNIST datasets gave different results, whereby the non-permutated dataset perfomed slightly better with a total accuracy of  98.392 % after 30 epochs vs. 95.990 % in the permutated dataset. Compare also the confusion matrices below.
These differences stem from stochastic effects, wherefore the pooled results of several initial runs are preferred over single runs alone.


## Without permutation of the data
![Confusion matrix test](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/CNN_30_0.1_test_confusion_matrix_29.png)

![Confusion matrix validation](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2c/figures/CNN_30_0.1_val_confusion_matrix_29.png)


## With permutation of the data
![Confusion matrix test permutated](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2d/figures/CNN_permuted_30.0.1_test_confusion_matrix_29.png)

![Confusion matrix validation permutated](https://github.com/pinargoktepe/PatRec/blob/master/Exercise_2d/figures/CNN_permuted_30_0.1_val_confusion_matrix_29.png)
