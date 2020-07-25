# Sentinel-NLP-Role-Tasks
This repository contains sources codes that are asked for a NLP role at Sentinel. 

The codes are compiled with python3.7. Virtualenv is utilized. 

Implemented models for task 1 is:
    - BERT
    - [NELA](https://github.com/BenjaminDHorne/The-NELA-Toolkit) feature based model either SVM or Random Forest
    - Simple ngrams either SVM or Random Forest
    
## Install

To initialize virtual environment, run the following code:
    `virtualenv -p /usr/bin/python3.7 venv`
    
## Task 1
Train set is split by sources. I ensure that train an dev set has different sources. However, split function gives each time different sets even random seed is set. Therefore I saved splits under `/data`. 

BERT is trained with GPU.

## Task 2

## Examples
Folder `/scripts` contains bash commands for each task.

## Acknowledgement
BERT is modified by the [following code](https://github.com/isspek/west_iyte_plausability_news_detection)
