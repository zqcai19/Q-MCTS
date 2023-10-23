# Q-MCTS
This repository contains the code for paper, Recognizing Good Variational Quantum Circuits with Monte Carlo Tree Search
```
requirements:

numpy
scikit-learn
pytorch
pennylane
```
To reproduce the results of the discovered best-performing hybrid fusion network which achieves MAE of 1.138 on MOSI dataset, run ```schemes.py``` in the folder ```best_vqc```.
The file ```Multibench.py``` contains the benchmark from <a href="https://github.com/pliang279/MultiBench/blob/main/examples/Multibench_Example_Usage_Colab.ipynb">MultiBench example</a> </br> where the dataset is changed to ours for fair comparison.
To start a search for hybrid architectures for multimodal task of sentiment analysis, simply run ```MCTS.py```.