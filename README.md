# Respiratory Disease Detection Using Audio Files

This repository contains code for detecting respiratory diseases from audio recordings using machine learning techniques. The system processes audio files and diagnosis data, extracts features using MFCCs (Mel-frequency cepstral coefficients), and classifies the disease using a convolutional neural network (CNN).

## Project Overview

The project aims to predict respiratory diseases such as URTI (Upper Respiratory Tract Infection) using sound recordings of patients' breathing. The audio data is processed to extract relevant features, which are then used to train a deep learning model.

## Features

- **Audio Feature Extraction:** Uses the `librosa` library to extract MFCC features from audio files.
- **Deep Learning Model:** A CNN-based model implemented using TensorFlow/Keras for classifying respiratory diseases.
- **File Handling:** Automatic matching of audio files with corresponding patient IDs from the diagnosis dataset.
- **Single-File Prediction:** Supports prediction of respiratory disease from a single audio file.

## Prerequisites

To run this project, you need the following dependencies:

- Python 3.x
- pandas
- numpy
- librosa
- scikit-learn
- TensorFlow/Keras

You can install the required libraries using:

```bash
pip install pandas numpy librosa scikit-learn tensorflow
