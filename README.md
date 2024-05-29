# tflite_micro_runtime
tflite_micro_runtime
for Raspberry PI Zero.

Prepare new Raspberry Pi Zero whit Raspberry Pi OS Lite

sudo apt-get update

sudo apt-get upgrade

reboot

sudo apt-get install libopenblas-dev

sudo apt-get install python3-pip

python3 -m venv --system-site-packages .venv

source .venv/bin/activate

pip3 install tflite_micro_runtime-1.2.2-cp311-cp311-linux_armv6l.whl

deactivate

--------------------------

For image recognition:

sudo apt-get install libopenjp2-7

source .venv/bin/activate

pip3 install numpy

pip3 install pillow

pip3 install argparse

deactivate

source test_image_recognition.sh

------------------------------------

For speech recognition:

sudo apt-get install libportaudio2

source .venv/bin/activate

pip3 install scipy

pip3 install python_speech_features

pip3 install sounddevice

pip3 install argparse

deactivate

attach USB microphone/USB webcam with microphone/USB sound card with microphone | connect BT hands free

source test_speech_wake_word.sh

------------------------------------

For recognition from camera:

sudo apt install -y python3-picamera2 --no-install-recommends

source .venv/bin/activate

pip3 install pillow

pip3 install argparse

deactivate

source test_notperson_person_camera.sh

.

tflite_micro_runtime binary code from https://github.com/driedler/tflite_micro_runtime
