Ce repo contient les projets implémentés pendant le cours [Deep Learning de A à Z](https://www.udemy.com/le-deep-learning-de-a-a-z/?couponCode=WEBSITE) à l'aide de `tensorflow`, `keras`, et `PyTorch`.

1. [Installation des modules](#installation-des-modules)
2. [Partie 1 - ANN](#partie-1-ann)
3. [Partie 2 - CNN](#partie-2-cnn)
4. [Partie 3 - RNN](#partie-3-rnn)
5. [Partie 4 - SOM](#partie-4-som)
6. [Partie 5 - BM](#partie-5-bm)
7. [Partie 6 - AE](#partie-6-ae)
8. [F.A.Q.](#faq)
    1. [Comment utiliser le GPU avec Tensorflow ?](#comment-utiliser-le-gpu-avec-tensorflow-)

## Installation des modules

Après avoir installé [Anaconda](https://anaconda.org/), suivre les instructions suivantes.

### Sur MacOS

```
conda create --name deeplearningaz python=3.6 anaconda
source activate deeplearningaz
conda install theano
conda install tensorflow
conda install keras
conda update --all
```

**Important** : Après avoir installé les modules, il faut toujours se remettre dans l'environnement créé (qu'on a appelé `deeplearningaz`) à l'aide de la commande `source activate deeplearningaz` avant de lancer Spyder. Sans quoi Spyder se lancera dans l'environnement par défaut.

### Sur Windows

```
conda create --name deeplearningaz python=3.6 anaconda
activate deeplearningaz
conda install theano
conda install tensorflow
conda install keras
conda update --all
```

**Important** : Après avoir installé les modules, il faut toujours se remettre dans l'environnement créé (qu'on a appelé `deeplearningaz`) à l'aide de la commande `activate deeplearningaz` avant de lancer Spyder. Sans quoi Spyder se lancera dans l'environnement par défaut.

### Sur Ubuntu

```
conda create --name deeplearningaz python=3.6 anaconda
source activate deeplearningaz
conda install theano
conda install tensorflow
conda install keras
conda update --all
```

**Important** : Après avoir installé les modules, il faut toujours se remettre dans l'environnement créé (qu'on a appelé `deeplearningaz`) à l'aide de la commande `source activate deeplearningaz` avant de lancer Spyder. Sans quoi Spyder se lancera dans l'environnement par défaut.

## Partie 1 - ANN

Le premier projet utilise un réseau de neurones artificiel utilisé pour prédire la probabilité de churn d'un panel de clients.

**Lectures additionnelles :**

* Yann LeCun et al., 1998, [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
* By Xavier Glorot et al., 2011, [Deep sparse rectifier neural networks](http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf)
* CrossValidated, 2015, [A list of cost functions used in neural networks, alongside applications](http://stats.stackexchange.com/questions/154879/a-list-of-cost-functions-used-in-neural-networks-alongside-applications)
* Andrew Trask, 2015, [A Neural Network in 13 lines of Python (Part 2 – Gradient Descent)](https://iamtrask.github.io/2015/07/27/python-network-part2/)
* Michael Nielsen, 2015, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)

## Partie 2 - CNN

Le deuxième projet propose l'utilisation d'un réseau de neurones à convolution pour classifier des images de chats et de chiens.

* Yann LeCun et al., 1998, [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
* Jianxin Wu, 2017, [Introduction to Convolutional Neural Networks](http://cs.nju.edu.cn/wujx/paper/CNN.pdf)
* C.-C. Jay Kuo, 2016, [Understanding Convolutional Neural Networks with A Mathematical Model](https://arxiv.org/pdf/1609.04112.pdf)
* Kaiming He et al., 2015, [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852.pdf)
* Dominik Scherer et al., 2010, [Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf)
* Adit Deshpande, 2016, [The 9 Deep Learning Papers You Need To Know About (Understanding CNNs Part 3)](https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html)
* Rob DiPietro, 2016, [A Friendly Introduction to Cross-Entropy Loss](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)
* Peter Roelants, 2016, [How to implement a neural network Intermezzo 2](http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/)

## Partie 3 - RNN

Le troisième projet vous apprend comment prédire la direction du prix d'une action grace aux réseaux de neurones récurrents.

* Oscar Sharp & Benjamin, 2016, [Sunspring](https://arstechnica.com/the-multiverse/2016/06/an-ai-wrote-this-movie-and-its-strangely-moving/)
* Sepp (Josef) Hochreiter, 1991, [Untersuchungen zu dynamischen neuronalen Netzen](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)
* Yoshua Bengio, 1994, [Learning Long-Term Dependencies with Gradient Descent is Difficult](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf)
* Razvan Pascanu, 2013, [On the difficulty of training recurrent neural networks](http://www.jmlr.org/proceedings/papers/v28/pascanu13.pdf)
* Sepp Hochreiter & Jurgen Schmidhuber, 1997, [Long Short-Term Memory](http://www.bioinf.jku.at/publications/older/2604.pdf)
* Christopher Olah, 2015, [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* Shi Yan, 2016, [Understanding LSTM and its diagrams](https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714)
* Andrej Karpathy, 2015, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
* Andrej Karpathy, 2015, [Visualizing and Understanding Recurrent Networks](https://arxiv.org/pdf/1506.02078.pdf)
* Klaus Greff, 2015, [LSTM: A Search Space Odyssey](https://arxiv.org/pdf/1503.04069.pdf)
* Xavier Glorot, 2011, [Deep sparse rectifier neural networks](http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf)

## Partie 4 - SOM

Le quatrième projet introduit les cartes auto-adaptatives pour détecter la fraude.

* Tuevo Kohonen, 1990, [The Self-Organizing Map](http://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf)
* Mat Buckland, 2004?, [Kohonen's Self Organizing Feature Maps](http://www.ai-junkie.com/ann/som/som1.html)
* Nadieh Bremer, 2003, [SOM – Creating hexagonal heatmaps with D3.js](https://www.visualcinnamon.com/2013/07/self-organizing-maps-creating-hexagonal.html)

## Partie 5 - BM

Le cinquième projet utilise les Machines de Boltzmann pour créer un système de recommandation

* Yann LeCun, 2006, [A Tutorial on Energy-Based Learning](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)
* Jaco Van Dormael, 2009, [Mr. Nobody](http://www.imdb.com/title/tt0485947/)
* Geoffrey Hinton, 2006, [A fast learning algorithm for deep belief nets](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)
* Oliver Woodford, 2012?, [Notes on Contrastive Divergence](http://www.robots.ox.ac.uk/~ojw/files/NotesOnCD.pdf)
* Yoshua Bengio, 2006, [Greedy Layer-Wise Training of Deep Networks](http://www.iro.umontreal.ca/~lisa/pointeurs/BengioNips2006All.pdf)
* Geoffrey Hinton, 1995, [The wake-sleep algorithm for unsupervised neural networks](http://www.gatsby.ucl.ac.uk/~dayan/papers/hdfn95.pdf)
* Ruslan Salakhutdinov, 2009?, [Deep Boltzmann Machines](http://www.utstat.toronto.edu/~rsalakhu/papers/dbm.pdf)

## Partie 6 - AE

Le sixième projet utilise les auto-encodeurs empilés, une technique avancée utilisée lors de la compétition Netflix.

* Malte Skarupke, 2016, [Neural Networks Are Impressively Good At Compression](https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/)
* Francois Chollet, 2016, [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
* Chris McCormick, 2014, [Deep Learning Tutorial – Sparse Autoencoder](http://mccormickml.com/2014/05/30/deep-learning-tutorial-sparse-autoencoder/)
* Eric Wilkinson, 2014, [Deep Learning: Sparse Autoencoders](http://www.ericlwilkinson.com/blog/2014/11/19/deep-learning-sparse-autoencoders)
* Alireza Makhzani, 2014, (k-Sparse Autoencoders](https://arxiv.org/pdf/1312.5663.pdf)
* Pascal Vincent, 2008, [Extracting and Composing Robust Features with Denoising Autoencoders](http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)
* Salah Rifai, 2011, [Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf)
* Pascal Vincent, 2010, [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf)
* Geoffrey Hinton, 2006, [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/science.pdf)

## F.A.Q.

### Comment utiliser le GPU avec Tensorflow ?

Par défaut, l'installation de Tensorflow utilise le CPU. Si vous avez un GPU puissant et souhaitez l'utiliser pour accélérer les calculs, il faut d'abord désinstaller la version CPU de tensorflow :

```
source activate deeplearningaz
conda remove tensorflow
```

Puis installer la version GPU. Anaconda se charge automatiquement d'installer les dépendances avec CUDA :

```
conda install tensorflow-gpu
conda install keras
```
