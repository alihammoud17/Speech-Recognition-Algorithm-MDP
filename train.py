import json
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001
""" epochs: it tells how many times the network is going 
to see the whole dataset for training purposes """
EPOCHS = 80
""" batch size: number of samples that the network will 
see before an update and running like the back propagation algorithm 
(learning SAP) """
BATCH_SIZE = 32
NUM_KEYWORDS = 4
PATIENCE = 20


def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    # extract inputs and targets
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Loaded training sets!")
    return X, y


def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    # load the dataset
    X, y = load_dataset(data_path)

    # create the train validation and test splits (2D arrays, coming from data.json)
    """test size: proportion of the dataset that's going to be
    allocated to the test set """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # reuse train test split for dividing the training set into training and validation
    """ test validation: proportion of the training sets that is going to be
     used for validation purposes """
    X_train, X_validation, \
    y_train, y_validation = train_test_split(X_train, y_train,
                                             test_size=test_validation)

    # convert inputs from 2D array to 3D arrays
    # (number of segments, 13(MFCCs))
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    # build the network
    model = keras.models.Sequential()

    # Convolutional neural network
    # conv layer 1
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  input_shape=input_shape))
    """kernel_regularizer=keras.regularizers.l2(0.001)))"""
    """ batch normalization: technique to speed up training 
    and get better results. It normalizes the activation in a 
     current layer """
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu"))
    """, kernel_regularizer=keras.regularizers.l2(0.001)))"""
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))
    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu"))
    """, kernel_regularizer=keras.regularizers.l2(0.001)))"""
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))
    # flatten the output, feed it into a dense layer
    """ Dense layers needs 1D array as input """
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))
    # softmax classifier, output layer
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))  # probabilities: ex: [0.1, 0.7, 0.1, 0.2]

    # compile the model (required by keras)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model
    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set
    :return history: Training history
    """

    # earlystop_callback = keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
                        verbose=1)
    # callbacks=[earlystop_callback])
    return history


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
    :param history: Training history of model
    :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()


def main():
    # load train/validation/test data splits
    # I/O target labels
    X_train, y_train, X_validation, y_validation, X_test, y_test = get_data_splits(DATA_PATH)

    # build CNN model
    """input_shape: it's the shape of 
    the input data that we are feeding into our CNN"""
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    # (number of segments(sr/hop_length), nb_coefficients(MFCCs), 1(carries infos about the depth ))
    """ we need a 3D array, because we are 
    dealing with a CNN that takes a 3D input """
    model = build_model(input_shape, LEARNING_RATE)

    # train model
    # method from keras api
    # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate model
    # test_error, test_accuracy = model.evaluate(X_test, y_test)
    # print(f"Test error: {test_error}, test accuracy: {test_accuracy * 100}")
    # predictions = model.predict_classes(X_test)
    # print(classification_report(y_test, predictions))
    # print(accuracy_score(y_test, predictions))

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
