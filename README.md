# Text Classifier with Deep Learning

This project demonstrates text classification using deep learning on 'fetch_20newsgroups' dataset of sci-kit learn.

It essentially classifies the news article to one of the target topics depending on the words and sequences used in the text.

It is a good practice and has explanations at every step.


We implement our model as a Sequential model from Keras. 

On Sequential models, layers are added one after another.

![Ekran görüntüsü 2024-10-24 190139](https://github.com/user-attachments/assets/13b9ac8d-3fc4-47e6-b689-776eb48ac132)


Embedding()                     => Embedding Layer

GlobalAveragePooling1D()        => Reducing the dimensionality of the Embedding outputs by taking the average of each sequence

Dense(32, activation='relu')    => Fully connected layer with 32 unit with relu activation function

Dense(20, activation='softmax') => Final output layer with 20 units, one for each class & softmax act. func. for probability distribution.




After training our model for 20 epochs, which processes 128 samples at a time, we evaluate the perfomance on the test data:

![Ekran görüntüsü 2024-10-24 170403](https://github.com/user-attachments/assets/45f4521c-d34b-4a76-98d5-25e3faf73cb8)


And see that a test accuracy of 0.75. 

Higher accuracy => model is making more correct predictions


Then we plot the training history to visualize the models perfomance over the epochs to inspect overfitting or underfitting:

![Figure_1](https://github.com/user-attachments/assets/958d49e4-d4bd-4e04-bb99-bb2f4be5c87e)


Finally, we use the trained model to make predictions on 5 test texts. 

![Ekran görüntüsü 2024-10-24 170419](https://github.com/user-attachments/assets/a58a1fa2-7069-43aa-a50b-fe1466ec4568)


Our model correctly classified the topics of 3 of the 5 test texts.


