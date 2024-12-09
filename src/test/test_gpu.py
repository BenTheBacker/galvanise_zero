# gpu_test.py

import tensorflow as tf
from keras import backend as K
import numpy as np

def test_tensorflow_gpu():
    # List available physical devices
    from tensorflow.python.client import device_lib
    devices = device_lib.list_local_devices()
    print("TensorFlow Available Devices:")
    for device in devices:
        print(device)

    # Create a simple TensorFlow session and perform a computation
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Define a simple computation graph
        a = tf.constant([1.0, 2.0, 3.0], name='a')
        b = tf.constant([4.0, 5.0, 6.0], name='b')
        c = a + b

        # Run the computation
        result = sess.run(c)
        print("TensorFlow Computation Result:", result)

def test_keras_gpu():
    # Check available GPUs via Keras backend
    gpus = K.tensorflow_backend._get_available_gpus()
    print("Keras Available GPUs:", gpus)

    # Create a simple Keras model and perform a forward pass
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(10, input_dim=3, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam')

    # Generate dummy data
    X = np.random.random((10, 3))
    y = np.random.randint(2, size=(10, 1))

    # Perform a single training step
    history = model.fit(X, y, epochs=1, batch_size=2, verbose=1)

if __name__ == "__main__":
    print("=== TensorFlow GPU Test ===")
    test_tensorflow_gpu()
    print("\n=== Keras GPU Test ===")
    test_keras_gpu()
