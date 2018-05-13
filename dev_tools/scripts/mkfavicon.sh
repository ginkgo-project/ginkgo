#!/bin/env bash

INPUT_IMAGE=$1
SIZES=256,128,64,48,32,16

convert ${INPUT_IMAGE} -define icon:auto-resize=$SIZES favicon.ico
