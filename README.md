# KGPC

--- PyTorch Version ---

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch 1.0 ](https://github.com/pytorch/pytorch) using [official website](https://pytorch.org/) or [Anaconda](https://www.continuum.io/downloads). 

2. Install the requirements: `pip install -r requirements.txt`

3. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`.

4. Download the data at https://dataoss.sctcc.cn/KGPC_data.zip

## Data Preprocessing

Run the preprocessing script for datasets: `python wrangle_KG.py patent569`.

## Run a model

To run a model, you first need to preprocess the data. 

The second is to change `os.environ['HOME']` to the absolute path of the project in `main.py`  
```
os.environ['HOME'] = '/python_work/KGPC/'
```


This can be done by specifying the `process` parameter.  
you can run it using:
```
python main.py --dataset patent569 --process True
```

You can modify the hyper-parameters from "config/*.yaml" or specify the hyper-parameters in the command.

For this test version, if you find any problems, please feel free and email me. We will keep updating the code.

