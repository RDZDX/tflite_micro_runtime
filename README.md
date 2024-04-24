# tflite_micro_runtime
tflite_micro_runtime
for Raspberry PI Zero.

Prepare new Raspberry Pi Zero whit Raspberry Pi OS Lite

sudo apt-get update

sudo apt-get upgrade

reboot

sudo apt-get install libopenblas-dev

sudo apt-get install python3-pip

python3 -m venv .venv

--------------------------

For image recognition:

sudo apt-get install libopenjp2-7

source .venv/bin/activate

pip3 install numpy

pip3 install pillow

pip3 install argparse

deactivate

------------------------------------

For speech recognition:

sudo apt-get install libportaudio2

source .venv/bin/activate

pip3 install scipy

pip3 install python_speech_features

pip3 install sounddevice

deactivate

Source from https://github.com/driedler/tflite_micro_runtime
