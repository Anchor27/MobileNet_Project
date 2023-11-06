<pre>
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Load and preprocess your data as needed
X_train = np.loadtxt('/content/drive/MyDrive/Image Classification CNN Keras Dataset/input.csv', delimiter=',')
Y_train = np.loadtxt('/content/drive/MyDrive/Image Classification CNN Keras Dataset/labels.csv', delimiter=',')
X_test = np.loadtxt('/content/drive/MyDrive/Image Classification CNN Keras Dataset/input_test.csv', delimiter=',')
Y_test = np.loadtxt('/content/drive/MyDrive/Image Classification CNN Keras Dataset/labels_test.csv', delimiter=',')

X_train = X_train.reshape(-1, 100, 100, 3)
X_test = X_test.reshape(-1, 100, 100, 3)

# EncodING class labels (0 for 'dog', 1 for 'cat')
Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

# Creating ImageDataGenerator for data augmentation and preprocessing
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

# Loading MobileNetV2 with pre-trained weights, excluding the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Adding custom top layers for binary classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Creating the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compiling the model with an appropriate optimizer and loss function
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model with data augmentation and save checkpoints
batch_size = 32
epochs = 5
model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=len(X_train) / batch_size, epochs=epochs)


# Model Evaluation

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')


# Generating a random index for a test image
random_idx = random.randint(0, len(Y_test) - 1)

# Displaying a random test image
plt.imshow((X_test[random_idx]* 255).astype('uint8'))
plt.show()

# Making predictions on the random test image
y_pred = model.predict(X_test[random_idx].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if y_pred == 0:
    pred = 'dog'
else:
    pred = 'cat'

print("Model Prediction:", pred)

from google.colab import drive
drive.mount('/content/drive')


model_save_path = '/content/drive/My Drive/Cat-Dog_Image_Classification_Model/'

# Save the entire model (architecture, weights, optimizer state)
model.save(model_save_path + 'Classification_Model.h5')

# If you only want to save the model's architecture and weights (not the optimizer state), you can use:
# model.save_weights(model_save_path + 'your_model_weights.h5')


!pip install scikit-learn
y_pred = model.predict(X_test)  

from sklearn.metrics import classification_report

y_pred_binary = (y_pred > 0.5).astype(int)

report = classification_report(Y_test, y_pred_binary)

# Print the report
print(report)

</pre>
