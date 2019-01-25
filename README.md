Ce repo contient les projets implémentés pendant le cours [Deep Learning de A à Z](https://www.udemy.com/le-deep-learning-de-a-a-z/?couponCode=WEBSITE) à l'aide de `tensorflow`, `keras`, et `PyTorch`.

## Installation des modules

Après avoir installé [Anaconda](https://anaconda.org/), suivre les instructions suivantes.

**Important** : Après avoir installé les modules, il faut toujours se remettre dans l'environnement créé (qu'on a appelé `deeplearningaz`) à l'aide de la commande `source activate deeplearningaz` avant de lancer Spyder. Sans quoi Spyder se lancera l'environnement par défaut.

### Sur MacOS

### Sur Windows

### Sur Ubuntu

```
conda create --name deeplearningaz python=3.6 anaconda
source activate deeplearningaz
conda install theano
conda install tensorflow
conda install keras
conda update --all
```

## Partie 1 - ANN

Le premier projet utilise un réseau de neurones artificiel utilisé pour prédire la probabilité de churn d'un panel de clients.
