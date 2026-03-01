"""Hand Sign Recognition Classifier using MobileNetV2."""

import glob
import os
import string

import cv2
import keras
import numpy as np

from pathlib import Path

english_alphabet = list(string.ascii_uppercase)
dict_classes = {label:character for label,character in enumerate(english_alphabet)}


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the dataset."""
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    data_dir = project_root / "data"
    
    X = []
    Y = []
    label = 0
    
    # Iterate through the data folder
    folders = sorted([f for f in data_dir.iterdir() if f.is_dir()])
    
    for folder in folders:
        for img_path in list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.resize(img, (224,224))
                X.append(img)
                Y.append(label)
        label += 1

    # Convert to numpy arrays
    X, Y = np.array(X), np.array(Y)

    # Shuffle the data
    indexes = np.random.permutation(len(X))
    X, Y = X[indexes], Y[indexes]

    # Split train/test 80/20
    split_ratio = int(len(X)*0.8)
    x_train, x_test = X[:split_ratio], X[split_ratio:]
    y_train, y_test = Y[:split_ratio], Y[split_ratio:]

    return x_train, y_train, x_test, y_test


def prepare_data(x_train:np.ndarray, y_train:np.ndarray, x_test:np.ndarray, y_test:np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], int]:
    """Reshape, normalize images and one-hot encode labels for model training."""
    # Reshape the data to be of size [samples][width][height][channels]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')
    input_shape = (x_train.shape[1], x_train.shape[2], 3)

    # Normalize the input values
    x_train = x_train / 255
    x_test = x_test / 255

    # Transform the classes labels into a binary matrix
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    num_classes = y_train.shape[1]

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def cnn_model(input_shape:tuple[int, int, int], num_classes:int) -> keras.models.Sequential:
    """Build and compile the model."""
    # Load the pretrained MobileNetV2 architecture and freeze it
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        classes=num_classes,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # Initialize the model
    model = keras.Sequential()
        
    # Add architecture
    model.add(keras.Input(shape=input_shape))
    model.add(base_model)
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.3))
        
    # Add the output layer
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=keras.optimizers.Adam(learning_rate=0.01), 
        metrics=['accuracy']
    )

    return model


def evaluate_on_test(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, input_shape:tuple[int, int, int], num_classes: int,) -> keras.models.Sequential:
    """Train the CNN on the full training set and evaluate accuracy on the test set."""
    # Build the model
    model = cnn_model(input_shape, num_classes)

    # Train on the full training dataset
    model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1)

    # Evaluate on the test dataset
    scores = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test accuracy: {scores[1]:.4f}")
    return model


def main():
    """Run the model on the loaded dataset."""
    # Load the dataset
    x_train, y_train, x_test, y_test = load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)

    # Evaluate the model
    model = evaluate_on_test(x_train, y_train, x_test, y_test, input_shape, num_classes)

    # Save the model
    model.save("data/model.keras")


if __name__ == "__main__":
    main()