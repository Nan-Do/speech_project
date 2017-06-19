#! /bin/bash

# Generate the virtualenv
virtualenv speech_project

# Copy the required files
cp speech_recognizer.py speech_project
cp -r audio speech_project

# Activate the virtual environment
. speech_project/bin/activate

# Install the different required packages
pip install numpy scipy
pip install pillow
pip install h5py
pip install tensorflow
pip install keras
pip install speechpy
