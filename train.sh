#!/usr/bin/env bash

clear

case "$1" in
    0)
    echo "run vgg"
    CUDA_VISIBLE_DEVICES=1 python train.py
    ;;

    *)
    echo
    echo "No input"
    ;;
esac
