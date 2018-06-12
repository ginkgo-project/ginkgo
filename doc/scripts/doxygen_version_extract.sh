#!/bin/bash
git log --format="format:%h" -n1 -- $1
