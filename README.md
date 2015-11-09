# tensorflow-testprojects

this repository is for testing some of the functionality of [tensorflow](http://www.tensorflow.org)

Created by Carl R. Knox on Mon Nov  9 10:43:08 MST 2015

## list of python modules in this repository
----

### basic-graph-building

* shows construction of a basic graph
* shows execution of ops in a graph (e.g. matrix multiplication op-type here)
* shows launching of a graph in a session
* shows use of tensorflow.device to utilize specific cores for ops

### basic-variables

* shows creation of a scalar variable and operations on it as a tensor, using tensorflow

### basic-fetches

* shows how to fetch multiple outputs of ops so that all ops are run once, not once per requested tensor.

### basic-feeds

* shows how to patch tensors directly into any op in a graph

----
## notes

* the module scripts are interpreted using a base tensorflow virtualenv, installation instructions [here](http://tensorflow.org/get_started/os_setup.md#virtualenv-based_installation)
* some IDE settings are included in this repo. if you don't want them then just delete the .idea directory
  from project root.
  
----
## references
* [tensorflow tutorials](http://tensorflow.org/tutorials)
* [MNIST Data](http://yann.lecun.com/exdb/mnist/)
* [MNIST Visualization](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)
* [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
* [Softmax Function Description](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)
* [Cross-Entropy](http://colah.github.io/posts/2015-09-Visual-Information/)