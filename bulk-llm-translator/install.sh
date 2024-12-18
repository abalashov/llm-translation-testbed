#!/bin/bash

# Initialise a new Python venv and install the various LLM provider SDKs there.
python3 -m venv our-env

source our-env/bin/activate
pip install --upgrade pip
pip install openai
exit