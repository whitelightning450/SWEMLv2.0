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

    mamba create -n SWEML_env python=3.9.12

For this example, we will be using Python version 3.9.12, specify this version when setting up your new virtual environment.
After Anaconda finishes setting up your SWEML_env , activate it using the activate function.

    mamba activate SWEML_env 

You should now be working in your new SWEML_env within the command prompt. 
However, we will want to work in this environment within our Jupyter Notebook and need to create a kernel to connect them.
We begin by installing the **ipykernel** python package:

    pip install --user ipykernel

With the package installed, we can connect the SWEML_env to our Python Notebook

    python -m ipykernel install --user --name=SWEML_env 

Under contributors, there is a start to finish example to get participants up to speed on the modeling workflow.
To double check you have the correct working environment, open the  up the [Methods](./contributors/NSM_Example/methods.ipynb) file, click the kernel tab on the top toolbar, and select the SWEML_env. 
The SWEML_env should show up on the top right of the Jupyter Notebook.


### Loading other Python dependencies
We will now be installing the packages needed to use SWEML_env, as well as other tools to accomplish data science tasks.
Enter the following code block in your terminal to get the required dependencies with the appropriate versions, note, you must be in the correct working directory:

    mamba env create -f SWEML_env2.yaml 

### Connect to AWS
All of the data for the project is on a publicly accessible AWS S3 bucket (national-snow-model), however, some methods require credentials. 
Please request credentials as an issue and put the credentials in the head of the repo (e.g., SWEMLv2.0) as AWSaccessKeys.csv.

