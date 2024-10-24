import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib. pyplot as plt

# Loading the 20 newsgroups dataset 
newsgroups = fetch_20newsgroups(subset='all')

# Extracting the text and labels
texts = newsgroups.data
labels = newsgroups.target

# Tokenizing the text
tokenizer = Tokenizer(num_words=10000)  # Tokenizer will only consider the most frequent 10000 words in the text data
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding the sequences
max_length = 100
data = pad_sequences(sequences, maxlen=max_length)  # Sequences shorter than the max length will be padded with 0's at the end

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)    # Transforming text labels into numerical representation

# Train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Building model
model = Sequential([    # On Sequential models, layers are added one after another
    Embedding(input_dim=10000, output_dim=32, input_length=max_length), # 
    GlobalAveragePooling1D(),   # Reducing the dimensionality of the Embedding outputs by taking the average of each sequence
    Dense(32, activation='relu'),   # Fully connected layer with 32 unit with relu activation function
    Dense(20, activation='softmax') # Final output layer with 20 units, one for each class & softmax act. func. for probability distribution
])

# Use 'adam' optimizer for updating weights during training 
# Sparse categorical cross entropy suitable for multiclass classification
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model for 20 epochs, processes 128 samples at a time during training
history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

# Evaluating the performence on test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')    # Higher accuracy => model is making more correct predictions

# Plotting training history to visualize models performance over the epochs
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Finally, using the trained model to make predictions on some test texts

predictions = model.predict(X_test[40:45])  # Makes predictionson the test texts from index 40 to 44
predicted_labels = [label_encoder.inverse_transform([tf.argmax(prediction).numpy()]) for prediction in predictions]

class_names = newsgroups.target_names

predicted_labels_names = [class_names[label[0]] for label in predicted_labels]  # Convert predicted labels to class names using class_names

acutal_labels = y_test[40:45]   

actual_label_names = [class_names[label] for label in acutal_labels]    # Convert actual labels to class names using class_names

# Display the texts, actual labels to class names using class_names
for i in range(5):
    #print(f'Texts {i+1}: {newsgroups.data[i]}')     # original content, comment it out to compare the results better
    print(f'Acutal Label: {actual_label_names[i]}')
    print(f'Predicted Label: {predicted_labels_names[i]}')
    print('========================')





