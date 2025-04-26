#!/bin/bash

echo 'removing __pycache__ directories ... '
find . -type d -name "__pycache__" -exec rm -rf {} \;

echo 'removing .DS_Store files ... '
find . -type f -name ".DS_Store" -exec rm -f {} \;
