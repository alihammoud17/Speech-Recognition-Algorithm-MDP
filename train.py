import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001
""" epochs: it tells how many times the network is going 
to see the whole dataset for training purposes """
EPOCHS = 40
""" batch size: number of samples that the network will 
see before an update and running like the back propagation algorithm 
(learning SAP) """
BATCH_SIZE = 32
NUM_KEYWORDS = 4


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs and targets
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Loaded training sets!")
    return X, y


def get_data_splits(data_path, test_size=0.2, test_validation=0.2):
    # load the dataset
    X, y = load_dataset(data_path)

    # create the train validation and test splits (2D arrays, coming from data.json)
    """test size: proportion of the dataset that's going to be
    allocated to the test set """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # reuse train test split for dividing the training set into training and validation
    """ test validation: proportion of the training sets that is going to be
     used for validation purposes """
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=test_validation)

    # convert inputs from 2D array to 3D arrays
    # (number of segments, 13(MFCCs))
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    # build the network
    model = keras.models.Sequential()

    # Convolutional neural network
    # conv layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu",
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    """ batch normalization: technique to speed up training 
    and get better results """
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    # flatten the output, feed it into a dense layer
    """ Dense layers needs 1D array as input """
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))  # probabilities: ex: [0.1, 0.7, 0.1, 0.2]

    # compile the model (required by keras)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model


def main():
    # load train/validation/test data splits
    # I/O target labels
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # build CNN model
    """input_shape: it's the shape of 
    the input data that we are feeding into our CNN"""
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # (number of segments(sr/hop_length), nb_coefficients(MFCCs), 1(carries infos about the depth ))
    """ we need a 3D array, because we are 
    dealing with a CNN that takes a 3D input """
    model = build_model(input_shape, LEARNING_RATE)

    # train model
    # method from keras api
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))

    # evaluate model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy * 100}")

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
