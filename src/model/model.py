import keras
import cv2
import numpy as np
import string
import glob
import os

english_alphabet = list(string.ascii_uppercase)
dict_classes = {label:character for label,character in enumerate(english_alphabet)}


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


def load_data():
    X = []
    Y = []
    label = 0

    for folder in sorted(glob.glob("data/*/")):
        for img_path in glob.glob(os.path.join(folder, "*.jpg")) + \
            glob.glob(os.path.join(folder, "*.png")) + \
            glob.glob(os.path.join(folder, "*.jpeg")):
            img = cv2.imread(img_path)
            if img is not None:
                X.append(img)
                X.append(label)
        label += 1

    X, Y = np.array(X, np.array(Y))

    # Shuffle
    indexes = np.random.permutation(len(X))
    X, Y = X[indexes], Y[indexes]

    # Split train/test 80/20
    split_ratio = int(len(X)*0.8)
    x_train, x_test = X[:split_ratio], X[split_ratio:]
    y_train, y_test = Y[:split_ratio], Y[split_ratio:]

    return x_train, y_train, x_test, y_test


def main():
    # Load the dataset
    x_train, y_train, x_test, y_test = load_data()

    # Preprocess the data
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)



if __name__ == "__main__":
    main()