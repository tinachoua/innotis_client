#!/bin/bash

conda activate innotis

if [[ $(basename $(pwd)) = "docker" ]];then
	cd ..
fi

python3 app.py
