import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

# Load in the mnist data set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# *Flatten basically creates an input layer for the NN
# *Dense is the traditionaly fully connected hidden layer of a NN
# *Dropout will create a layer that randomly sets some of its inputs to 0 at 
#  some rate during training which helps prevent overfitting. This layer does
#  not randomly set inputs to zero during evaluation. 

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28),  dtype='float32'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# Apply the model to the first n sample, in this case n = 1
predictions = model(x_train[:1]).numpy()

# Apply softmax to the predictions to convert them into effective probabilities 
tf.nn.softmax(predictions).numpy()

# Creating the loss function to optimize
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Minimizing the loss function using the adam optimizer
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the NN to minimize the loss function
model.fit(x_train, y_train, epochs=2)

# Evaluate the model on the test set
model.evaluate(x_test,  y_test, verbose=2)

# Adding a softmax layer to the end of the already trained network
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])


testImageNum = np.round(np.random.rand(3,3)*y_test.shape[0])


fig, axs = plt.subplots(3,3)

for n in range(3):
    for m in range(3):
        testPred = probability_model(x_test[int(testImageNum[n,m]), :, :].reshape(1,28,28))
        
        predictedDigit = np.argmax(testPred);
        Prob = testPred[0].numpy()[predictedDigit];
        
        testImage = np.squeeze(x_test[int(testImageNum[n,m]), :,:])
        axs[n,m].imshow(testImage, cmap='gray')
        axs[n,m].set_title('Digit is a {DigitVal} with {P:.2f}% confidence'\
                  .format(DigitVal = predictedDigit, P = Prob*100))
    
fig.tight_layout()
plt.show()