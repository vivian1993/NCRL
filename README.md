# NCRL
## Introduction
The official Pytorch implementation of the paper Neural Compositional Rule Learning for Knowledge Graph Reasoning

## KG Data:
* entities.txt: a collection of entities in the KG
* relations.txt: a collection of relations in the KG
* facts.txt: a collection of facts in the KG 
* train.txt: the model is trained to fit the triples in this data set
* valid.txt: create a blank file if no validation data is available
* test.txt: the learned ryles is evaluated on this data set for KG completion task

## Usage
For example, this command train a NCRL on family dataset using gpu 0
```
  python main.py --train --test --data family --max_path_len 4 --model family --gpu 0 --get_rule --topk 500
```
Each parameter means:
* --train: train the model
* --test: assign score to each rule in the rule space
* --max_path_len: the maximum length of paths observed during training
* --get_rule: output the learned rules
* --data: dataset
* --topk: number of the output rules
* --model: where do we save our model
