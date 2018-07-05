#!/bin/bash
cd $1 && git log --format="format:%h" -n1 -- $2
