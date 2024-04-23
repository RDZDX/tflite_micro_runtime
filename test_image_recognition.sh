#!/bin/bash
if [[ "$0" = "$BASH_SOURCE" ]]; then
    echo "Needs to be run using source: source ${0##*/}"

else
    VENVPATH="/home/pi/.venv/bin/activate"
    if [[ $# -eq 1 ]]; then 
        if [ -d $1 ]; then
            VENVPATH="$1/bin/activate"
        else
            echo "Virtual environment $1 not found"
            return
        fi

    fi

    echo "Activating virtual environment $VENVPATH"
    source "$VENVPATH"

python3 label_image.py --model_file cifar_10.tflite --label_file labels_cifar_10.txt --image test_cat.jpg --input_mean 0 --input_std 255

deactivate
fi