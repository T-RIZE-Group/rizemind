#import os

import flwr as fl
import tensorflow as tf


# Make TensorFlow log less verbose
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
#input shape
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
#compile the model with an optimizer. So, this is just defining the loss function
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#Flower provides a convenience class called NumPyClient which makes it easier to implement the Client

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}
    
#create an instance of our class CifarClient and add one line to actually run this client

fl.client.start_client(server_address="127.0.0.1:8080", client=CifarClient().to_client())
#start_numpy_client()
#fl.client.start_client(server_address="[::]:8080", client=CifarClient().to_client())
#fl.client.start_numpy_client("127.0.0.1:8081", client=CifarClient())