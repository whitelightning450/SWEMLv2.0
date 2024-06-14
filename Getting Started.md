![NSM_Cover](./Images/ML_SWE.jpg)

# Snow Water Equivalent Machine Learning (SWEMLv2.0): Using Machine Learning to Advance Snow State Modeling

# Getting Started: 
Please for this repo to your GitHub account.
Next, identify a folder location where you would like to work in a development environment.
Using the command prompt, change your working directory to this folder and git clone https://github.com/USERID/SWEMLv2.0

    git clone https://github.com/USERID/SWEMLv2.0


## Virtual Environment
It is a best practice to create a virtual environment when starting a new project, as a virtual environment essentially creates an isolated working copy of Python for a particular project. 
I.e., each environment can have its own dependencies or even its own Python versions.
Creating a Python virtual environment is useful if you need different versions of Python or packages for different projects.
Lastly, a virtual environment keeps things tidy, makes sure your main Python installation stays healthy and supports reproducible and open science.

## Creating Stable CONDA Environment on HPC platforms
Go to home directory
```
cd ~
```
Create a envs directory
```
mkdir envs
```
Create .condarc file and link it to a text file
```
touch .condarc

ln -s .condarc condarc.txt
```
Add the below lines to the condarc.txt file
```
# .condarc
envs_dirs:
 - ~/envs
```
Restart your server

### Creating your SWEML_env Virtual Environment
Since we will be using Jupyter Notebooks for this exercise, we will use the Anaconda command prompt to create our virtual environment. 
We suggest using Mamba rather than conda for installs, conda may be used but will take longer.
In the command line type: 

    mamba env create -f SWEML_310environment.yml 

    conda activate SWEML_310

    mamba install -c conda-forge boto3

    python -m ipykernel install --user --name=SWEML_310


### Connect to AWS
All of the data for the project is on a publicly accessible AWS S3 bucket (national-snow-model), however, some methods require credentials. 
Please request credentials as an issue and put the credentials in the head of the repo (e.g., SWEMLv2.0) as AWSaccessKeys.csv.

