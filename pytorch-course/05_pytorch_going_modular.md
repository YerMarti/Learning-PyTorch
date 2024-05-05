# PyTorch - Going Modular

In this lesson, we are going to convert our Jupyter Notebook / Google Collab Notebook into Python scripts. We are going to use the previous lesson's ([04 - PyTorch Custom Datasets](04_pytorch_custom_datasets.ipynb)) notebook and we are going to parse the workflow there (the fundamental code) to scripts.

## How are we doing it?

We are going to convert the code in the notebook cells and tweak it so we can structure it as Python scripts. The file structure could be something like this:

* `data_setup.py` - a file to prepare and download the data.
* `engine.py` - a file containing various functions.
* `model_builder.py` - a file to create the PyTorch model.
* `train.py` - a file to leverage all other files and train the model.
* `utils.py` - a file containing utility functions.

## Why should we go modular?

While notebooks are great to test and run experiments quickly, scripting is better for reproducing and running larger projects. Each one has its pros and cons.

### Notebooks

| **Pros** | **Cons** |
|:---:|:---:|
| Easy to experiment/get started | Versioning can be hard |
| Easy to share (e.g. a link to a Google Colab notebook) | Hard to use only specific parts |
| Very visual | Text and graphics can get in the way of code |

### Python scripts

| **Pros** | **Cons** |
|:---:|:---:|
| Can package code together (saves rewriting similar code across different notebooks) | Experimenting isn't as visual (usually have to run the whole script rather than one cell) |
| Can use git for versioning | - |
| Many open source projects use scripts | - |
| Larger projects can be run on cloud vendors (not as much support for notebooks) | - |

## Common usage

It is very common for PyTorch-based Machine Learning projects to have instructions on how to run the code from the scripts via terminal / CLI. E.g:

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

These custom parameters are known as argument flags. We can set up any number of flags we need. Our goal is to replicate this with the scripts we are going to create for the previous notebook.

## 1. Get data