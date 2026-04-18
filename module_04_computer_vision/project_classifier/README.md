# CIFAR-10 Image Classification Project

A complete, well-structured image classification pipeline using CIFAR-10.
This project demonstrates proper separation of concerns in ML code.

## Structure
```
project_classifier/
├── data/dataset.py        — Data loading and augmentation
├── models/cnn.py          — Model architecture
├── training/train.py      — Training pipeline
├── evaluation/evaluate.py — Evaluation and visualization
└── config.yaml            — Experiment configuration
```

## Usage
```bash
cd module_04_computer_vision/project_classifier
python training/train.py
python evaluation/evaluate.py
```
