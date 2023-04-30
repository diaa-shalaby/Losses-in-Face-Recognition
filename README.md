# Face Recognition and Verification using Facenet Model

This repository contains the code for a project that explores the effectiveness of the Facenet model for face recognition and verification tasks. The project focuses on the intermediate embeddings created by the Facenet model and investigates the impact of the triplet loss on model performance.

## Project Structure

The project is structured as follows:

- `lfw/` - `\lfw_cropped` : contains the Labeled Faces in the Wild (LFW) dataset used for training and evaluation.
- `models/`: contains the implementation of the backbone model i.e. ResNet, VGG in PyTorch.
- `train.py`: script for training the Facenet model with different loss functions (triplet loss and MSE loss).
- `evaluate.py`: script for evaluating the trained model on the LFW test set.
- `utils.py`: contains utility functions for loading and preprocessing the data.

## Getting Started

To run the code in this repository, you will need to have Python 3 and PyTorch installed on your system. You can install PyTorch using pip:

```
pip install torch torchvision
```

You will also need to download the LFW dataset and place it in the `data/` directory. The dataset can be downloaded from the official website: http://vis-www.cs.umass.edu/lfw/

Once you have the dataset and the necessary dependencies installed, you can train and evaluate the Facenet model by running the `train.py` and `evaluate.py` scripts, respectively:

```
python train.py
python evaluate.py
```

## Results

The results of the experiments are presented in the project report, which can be found in the `report/` directory.

## References

- [Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "FaceNet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.](https://arxiv.org/abs/1503.03832)
- [Labeled Faces in the Wild (LFW) dataset](http://vis-www.cs.umass.edu/lfw/)
