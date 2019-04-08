
# Permuted MNIST on MLP

First, let's compare results between MLP network trained on MNIST and one trained on Permutated MNIST:

There is no difference, we can see this from graph_1 and test_accuracies.

# Permuted MNIST on CNN

Now let's compare results between CNN network trained on MNIST and one trained on Permutated MNIST:

If we look at graph_2 and test_accuracies, obviously there is a difference now. CNN performs around 3 percents
better when is trained on MNIST than when it's trained with Permutated MNIST. Additionally CNN trained on 
Permutated MNIST performs slightly worse (less than a percent) than both MLP networks. This phenomenon 
can be explained as CNN networks use convolutional layers to search for patterns. While on normal MNIST 
dataset we have clear patterns as part or whole numbers in Permutated MNIST dataset only information from 
a picture is pixel values. Therefore MLP which only looks for pixel values performs the same on both datasets,
while CNN performs better on normal MNIST dataset, where it can learn additional information which is patterns 
of numbers.
