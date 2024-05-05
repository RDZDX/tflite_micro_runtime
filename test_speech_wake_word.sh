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

#    echo "Activating virtual environment $VENVPATH"
    source "$VENVPATH"

python3 wake_word_stop.py
deactivate
fi
