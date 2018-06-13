#!/bin/bash
cat $1 | perl -0777 -pe 's|<<<.*?>>>||smg'
