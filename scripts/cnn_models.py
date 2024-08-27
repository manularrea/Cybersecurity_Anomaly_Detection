import sys
import os
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.utils import to_categorical

# Ensure UTF-8 encoding for all output
sys.stdout.reconfigure(encoding='utf-8')

# Define CNN models with increasing complexity
def create_simple_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_intermediate_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_advanced_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_cnns(input_file, model_output_dir):
    df = pd.read_csv(input_file)

    # Separate features and labels
    X = df.drop(columns=['Label'])
    y = df['Label']

    # Ensure all data is numeric (convert boolean features to float)
    X = X.astype('float32')

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply undersampling to balance the classes
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y_encoded)

    # Use StratifiedShuffleSplit to split the data in a stratified manner
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in sss.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

    # Convert y_train and y_test to one-hot encoded format for CNN
    num_classes = len(label_encoder.classes_)
    y_train_categorical = to_categorical(y_train, num_classes=num_classes)
    y_test_categorical = to_categorical(y_test, num_classes=num_classes)

    # Reshape the data for CNN (assuming time-series data or 1D data)
    X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Define CNN models
    models = {
        "simple_cnn": create_simple_cnn(input_shape=(X_train.shape[1], 1), num_classes=num_classes),
        "intermediate_cnn": create_intermediate_cnn(input_shape=(X_train.shape[1], 1), num_classes=num_classes),
        "advanced_cnn": create_advanced_cnn(input_shape=(X_train.shape[1], 1), num_classes=num_classes)
    }

    # Train and save each model
    for name, model in models.items():
        model.fit(X_train_cnn, y_train_categorical, validation_split=0.1, epochs=10, batch_size=32, verbose=1)
        model.save(os.path.join(model_output_dir, f"{name}_model.h5"))
        print(f"{name} model saved.")

    # Save the label encoder for future use
    joblib.dump(label_encoder, os.path.join(model_output_dir, "label_encoder.pkl"))

if __name__ == "__main__":
    train_cnns("data/processed/engineered_data.csv", "data/models/")
