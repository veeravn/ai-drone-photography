#!/bin/bash
sudo apt update && sudo apt install -y python3-pip git
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
echo "✅ AI-Photographer dependencies installed"