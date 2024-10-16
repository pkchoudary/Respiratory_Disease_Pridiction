import os
import re
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

def load_data(audio_dataset_path, diagnosis_dataset_path):
    audio_files = []
    labels = []

    # Load diagnosis dataset
    diagnosis_data = pd.read_csv(diagnosis_dataset_path)
    print("Columns in CSV:", diagnosis_data.columns)  # Check columns in CSV

    # Print files in directory for debugging
    audio_files_in_dir = os.listdir(audio_dataset_path)
    print("Files in audio directory:", audio_files_in_dir)

    for index, row in diagnosis_data.iterrows():
        patient_id = str(row['101'])
        # Find the matching audio file using regex
        matched_files = [f for f in audio_files_in_dir if re.match(f"{patient_id}_.*\.wav", f)]

        if matched_files:
            audio_filename = matched_files[0]  # Use the first match found
            audio_path = os.path.join(audio_dataset_path, audio_filename)
            print(f"Using file: {audio_path}")  # Print path to debug
            audio_files.append(audio_path)
            labels.append(row['URTI'])  # Update with correct column name if different
        else:
            print(f"No matching file found for patient ID: {patient_id}")

    if not audio_files:
        print("No audio files found. Exiting.")
        exit()

    return audio_files, labels

def extract_features(file_paths):
    features = []

    for file in file_paths:
        try:
            y, sr = librosa.load(file, duration=30)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            features.append(mfccs_mean)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return np.array(features)

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def predict_single_file(model, file_path, scaler, label_encoder):
    # Extract features from the single audio file
    features = extract_features([file_path])

    # Normalize features
    features = scaler.transform(features)

    # Reshape features for the model
    features = np.expand_dims(features, axis=-1)

    # Predict the class
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)

    # Decode the label
    predicted_label = label_encoder.inverse_transform(predicted_class)

    return predicted_label[0]

def main():
    # Define paths
    audio_dataset_path = 'C:\\Users\\vpava\\Documents\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files'  # Directory containing audio files
    diagnosis_dataset_path = r'C:\Users\vpava\Documents\Respiratory_Sound_Database\Respiratory_Sound_Database\patient_diagnosis.csv'  # Update this path

    # Load data
    audio_files, labels = load_data(audio_dataset_path, diagnosis_dataset_path)

    # Extract features
    features = extract_features(audio_files)

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build model
    input_shape = (X_train.shape[1], 1)
    num_classes = len(np.unique(labels_encoded))
    model = build_model(input_shape, num_classes)

    # Train model
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
    X_test = np.expand_dims(X_test, axis=-1)  # Add channel dimension
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate model
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Predict on a single file
    single_file_path = 'C:\\Users\\vpava\\Documents\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files\\101_1b1_Al_sc_Meditron.wav'  # Path to the specific file
    predicted_label = predict_single_file(model, single_file_path, scaler, label_encoder)
    print(f"The predicted disease for the file {single_file_path} is: {predicted_label}")

if __name__ == '__main__':
    main()
