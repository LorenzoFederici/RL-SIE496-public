# REINFORCEMENT LEARNING FOR ROBOT CONTROL

This repository is meant for the students of Prof. Fabio Curti's Robotic Systems class (SIE 496/596), spring 2025, at the Systems & Industrial Engineering Department of the University of Arizona.
The aim of this repository is to show how to formulate your robot control problem as a Reinforcement Learning (RL) environment (i.e., Markov decision process) by exploiting the OpenAI [Gymnasium](https://gymnasium.farama.org/) Python library, and how to train a neural network to solve this problem via an RL algorithm using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html) Python library.

## INSTALLATION

The software runs on Linux (Ubuntu), macOS, and Windows (through Windows Subsystem for Linux, WSL). To correctly set up the software, please follow the present instructions:

1. First, you need Python3 and the system packages gcc, g++, and make:
    ```bash
    sudo apt-get update
    sudo apt-get upgrade
    sudo apt-get install python3-dev build-essential
    ```

2. Install [Anaconda](https://www.anaconda.com/distribution/) on Linux via command line with the command:
    ```bash
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh
    ./Anaconda3-2024.10-1-Linux-x86_64.sh
    ```
    Then, press enter until the installation process starts.

3. Now, use conda to create a virtual environment, named rl-env, with a specific Python version (3.10 and upper) by using conda:
    ```bash
    conda create -n rl-env python=3.10
    ```

3. After activating the environment with:
    ```bash
    conda activate rl-env
    ```
    you can install all required packages via pip:
    ```bash
    pip install tensorboard swig box2d pygame ale-py gymnasium[all] stable-baselines3
    ```

##  USAGE

To test an environment with a random agent, specify the environment name within the script `run_env.py`. Then, launch the script via:
```bash
python run_env.py
```

To train a new model via RL and save it, specify the environment name, model architecture, algorithm to use, and algorithm hyperparameters within the script `train.py` or `train_atari.py` (for [Atari](https://ale.farama.org/environments/) environments). Then, launch the script via:
```bash
python train.py
```
or
```bash
python train_atari.py
```

To load a pretrained model, evaluate its performance, and test it in deployment mode, specify the environment name within the script `evaluate.py` or `train_atari.py` (for [Atari](https://ale.farama.org/environments/) environments), and launch it via:
```bash
python evaluate.py
```
or
```bash
python evaluate_atari.py
```


