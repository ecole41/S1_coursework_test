

# Project Setup for Jupyter Notebook and Conda Environment

This repository contains a Jupyter notebook and Python scripts that require specific dependencies. To make it easy to replicate the environment, we use a Conda environment and include an `environment.yml` file to manage dependencies.

## Prerequisites

1. **Conda**: Ensure you have [Conda](https://docs.conda.io/en/latest/) installed. If you don't have Conda installed, you can download and install it from the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) or the [Anaconda website](https://www.anaconda.com/products/individual).

2. **Jupyter**: We'll use Jupyter Notebook for running the Python scripts and analysis.

## Setting up the Environment

Follow these steps to set up your Conda environment and run the Jupyter notebook:

### Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/s1_coursework/egc41
cd egc41
conda env create --file environment.yml
conda activate s1_env
```
### Step 2: Jupyter notebook
To open the directory in the form of a Jupyter Notebook run:

```bash
jupyter notebook
```
In this directory, the `Coursework_S1.ipynb` file contains all of the code needed to produces results. Please run this whole notebook.

### Running Python Scripts

The Python scripts `Parametric_bootstrapping.py` and `sWeighting.py` have already been run and their results saved in the `Parametric_Bootstrapping_Data` and `sWeighting_Data` folder respectively, so there is no need to run these again.

But if you wish to run the Python scripts to generate and analyse the Parametric Bootstrapping data and analyse using sWeights, then run these commands:

```bash
python Parametric_bootstrapping.py
```

and 

```bash
python sWeighting.py
```

These modules have been parallelized to improve efficiency.
