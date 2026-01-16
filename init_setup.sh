#!/bin/bash

echo [$(date)]: "START"

echo [$(date)]: "Creating conda env with Python 3.10 version"

conda create -p ./env python=3.10 -y

echo [$(date)]: "Activating the environment"

conda activate $(pwd)/env

echo [$(date)]: "Installing the dev requirements"

python -m pip install --upgrade pip

pip install -r requirements.txt 

echo [$(date)]: "Installing the project as local package"

pip install -e .

echo [$(date)]: "END"
