import random
import json
import pickle
import numpy as np
import nltk
import tensorflow as tf

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Define a function to load your existing dataset
def load_data():
    # You can load any dataset you want or create one here
    # For demonstration, we're creating a simple patterns list
    intents = {
        "intents": [
            {"tag": "greeting", "patterns": ["Hi", "Hello", "How are you?"], "responses": ["Hello!", "Hi there!", "Greetings!"]},
            {"tag": "goodbye", "patterns": ["Bye", "See you", "Goodbye"], "responses": ["Goodbye!", "See you later!"]},
            {"tag": "thanks", "patterns": ["Thanks", "Thank you", "That's helpful"], "responses": ["You're welcome!", "Happy to help!"]}
        ]
    }
    return intents

# Prepare training data
def prepare_data(intents):
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and filter words
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(set(words))
    classes = sorted(set(classes))

    # Save words and classes to files
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    return documents, words, classes

# Prepare training data for the model
def train_model(documents, words, classes):
    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append(bag + output_row)

    random.shuffle(training)
    training = np.array(training)

    # Split into inputs and outputs
    train_x = training[:, :len(words)]
    train_y = training[:, len(words):]

    # Build the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

    # Compile the model
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Train the model
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    # Save the trained model
    model.save('chatbot_model.keras')
    print('Model training complete and saved.')

if __name__ == "__main__":
    intents = load_data()
    documents, words, classes = prepare_data(intents)
    train_model(documents, words, classes)
