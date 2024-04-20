#!/bin/bash
if [[ "$0" = "$BASH_SOURCE" ]]; then
    echo "Needs to be run using source: 'source test.sh'"

else
    VENVPATH="/home/pi/.venv/bin/activate"
    if [[ $# -eq 1 ]]; then 
        if [ -d $1 ]; then
            VENVPATH="$1/bin/activate"
        else
            echo "Virtual environment $1 not found"
            return
        fi

    elif [ -d "venv" ]; then 
        VENVPATH="venv/bin/activate"

    elif [-d "env"]; then 
        VENVPATH="env/bin/activate"
    fi

    echo "Activating virtual environment $VENVPATH"
    source "$VENVPATH"

python3 label_image.py --model_file cifar_10.tflite --label_file labels_cifar_10.txt --image test_cat.jpg

deactivate
fi
