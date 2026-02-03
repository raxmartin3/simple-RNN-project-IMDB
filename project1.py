import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping


max_features = 10000
max_len = 200


(X_train, Y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(f"Training data shape: {len(X_train)}, Training label shape: {len(Y_train)}")
print(f"Test data shape: {len(x_test)}, Test label shape: {len(y_test)}")


x = np.concatenate((X_train, x_test), axis=0)
y = np.concatenate((Y_train, y_test), axis=0)

split_index = int(0.8 * len(x))

X_train = x[:split_index]
Y_train = y[:split_index]
x_test = x[split_index:]
y_test = y[split_index:]

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

print("Padded X_train shape:", X_train.shape)


model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128, input_length=max_len))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


earlystopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    Y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping]
)


accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(accuracy) + 1)

plt.figure(figsize=(7,4))
plt.plot(epochs, accuracy, marker='o', label='Training Accuracy')
plt.plot(epochs, val_accuracy, marker='s', label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(7,4))
plt.plot(epochs, loss, marker='o', label='Training Loss')
plt.plot(epochs, val_loss, marker='s', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


model.save("sample_rnn_imdb.h5")


model = load_model("sample_rnn_imdb.h5")
model.summary()


test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


sample_review = x_test[1]
print(sample_review)

len(sample_review.reshape(1,-1))
prediction= model.predict(sample_review.reshape(1,-1))
prediction
sentiment="positive" if  prediction[0][0]>0.5 else "negative"
print("predicted sentiment:", sentiment)
print("predicted score:", prediction[0][0])
print("actual label :", "positive" if y_test[0]==1 else "begative")

