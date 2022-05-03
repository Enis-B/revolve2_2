<img  align="right" width="150" height="150"  src="/docs/source/logo.png">

# Revolve2
Revolve2 is a Python package for optimization, geared towards modular robots and evolutionary computing.
Its primary features are a modular robot framework, wrappers around physics simulators, and evolutionary optimizers.
It consists of multiple smaller python packages and optional supplementary packages that contain varying functionality that is not always required.
  

## Prerequisites 

If you are using the Isaac Gym environment supplementary library, it requires exactly python 3.8. It's working on Linux and windows only. 

  

## Installation 
Assuming that you have downloaded isaacgym and installed virtualenv, simple steps to install are:
``` 
cd revolve2/
python3.8 -m virtualenv .venv
source .venv/bin/activate
pip install ~/isaacgym/python/
./dev_requirements.sh
``` 
check the installation
``` 
pip list |grep revolve2
``` 
## Step by step

1. Install virtual enviroment. 1) virtualenv or 2)conda.  

  

* if you choose virtualenv, install virtualenv: 

  

``` 

pip3.8 install virtualenv 

``` 

Create a directory for your project, then create a virtual environment: 

``` 

python3.8 -m virtualenv .venv 

``` 

Activate the virtual environment: 

``` 

source .venv/bin/activate 

``` 

* if you choose conda, install anaconda: 

Download [anaconda] package (https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) 

Go to the directory where you've stored the package (e.g. cd ~/Download) 

``` 

bash Anaconda-latest-Linux-x86_64.sh (make sure it's matching the pageckage name you downloaded) 

``` 

test if conda is installed correctly:  

``` 

conda list 

``` 

Activate the virtual environment: 

``` 

conda activate [virtual env name] 

``` 

if you don't know which virtual env to activate, check your virtual env list by  

``` 

conda info --env 

``` 

2. Clone the Revolve2 project locally: 

``` 

git clone https://github.com/ci-group/revolve2.git --recursive 

``` 

  

3. Install core 

Core has only PyPI dependencies and is a pure python library: 

``` 

pip install <revolve_path>/core 

``` 

4. Obtain a copy of Isaac Gym version rc3. At the time of writing it is available at https://developer.nvidia.com/isaac-gym.  

5. Next, make sure you are in your virtual environment and install Isaac Gym: 

``` 

pip install <isaacgym_path>/python 

``` 

6. Install the Isaacgym library as a python package: 

``` 

pip install <revolve_path>/runners/isaacgym 

``` 

7. Install boost (Ubuntu) 

``` 

sudo add-apt-repository universe 

sudo apt-get update 

sudo apt-get install libboost-all-dev 

``` 

8. Intall cppnwin 

``` 

pip install <revolve_path>/genotypes/cppnwin 

``` 

Ubuntu user: 

``` 

sudo apt install libcereal-dev 

``` 

## Testing 1

``` 

cd ~/isaacgym/python/examples/ 

python joint_monkey.py  

``` 

![isaacgym](https://user-images.githubusercontent.com/75667244/153434643-80b9317c-f41c-4508-8188-20bb4973d724.png) 

  

 ## Testing 2
 ``` 
cd ~/revolve2/examples/optimize_modular
python optimize.py
``` 
``` 
cd ~/revolve2/core/revolve2/analysis/core
python plot_ea_fitness.py ~/revolve2/examples/optimize_modular/database/

``` 
![image](https://user-images.githubusercontent.com/75667244/161257429-6b8fdba4-997e-44be-bb84-57662d79a9a5.png)

``` 
In the newest version: +database +process_id
~/revolve2$ revolve2_plot_ea_fitness_float ./examples/simple_optimization/database 0

``` 
![Figure_1](https://user-images.githubusercontent.com/75667244/162200509-5b341b1c-fa7c-4337-9087-dbcadc8f4632.png)

 ## Extra - if you'd like to rebase to the master branch of the original revolve2 version
1. Fork [Revolve2](https://github.com/ci-group/revolve2)
2. Add the forked repo as a submodule under your working direcotry. e.g.:
``` 
  git submodule add --force git@github.com:onerachel/revolve2.git
``` 
or checkout to the existing one:
``` 
  git checkout remote/xxx
``` 
3. Remote to the original revolve2
``` 
cd revolve2
git remote add ci_revolve2 git@github.com:ci-group/revolve2.git

``` 
Check if it's added by 
```
git remote -v
```
4. Get the content from the original CI revolve2
```
git fetch ci_revolve2
```
5. Rebase the master branch of the original revolve2 as the root
```
git rebase ci_revolve2/master
```
6. Push twice under revolve2 directory and your working directory
```
git push --force-with-lease
```
 
## Documentation 

[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/) 

 
>>>>>>> 3749374 (Update README.md)
