# GPolS: A Contextual Graph-Based Language Model for Analyzing Parliamentary Debates and Political Cohesion

This codebase contains the python scripts for GPolS, the model for the EMNLP 2020 paper [link](https://www.aclweb.org/anthology/2020.coling-main.426.pdf).

## Environment & Installation Steps
Python 3.6, Pytorch, and networkx


```python
pip install -r requirements.txt
```

## Dataset and Preprocessing 

Download the dataset from [here](https://data.mendeley.com/datasets/czjfwgs9tm/2). 

Follow [link](https://huggingface.co/transformers/training.html) to fine-tune BERT using speech transcripts on the classification task. Store the transcript and motion text features in folders "speeches/" and "motions/", respectively.


Generate graph

```python
python graph_make.py
```

Prepare model inputs and labels
```python
python preprocess.py
```

## Run

Execute the following python command to train GPolS: 
```python
python train.py
```

## Cite
Consider citing our work if you use our codebase

```c
@inproceedings{sawhney-etal-2020-gpols,
    title = "{GP}ol{S}: A Contextual Graph-Based Language Model for 
Analyzing Parliamentary Debates and Political Cohesion",
    author = "Sawhney, Ramit  and
      Wadhwa, Arnav  and
      Agarwal, Shivam  and
      Shah, Rajiv Ratn",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.426",
    doi = "10.18653/v1/2020.coling-main.426",
    pages = "4847--4859",
    }
```
