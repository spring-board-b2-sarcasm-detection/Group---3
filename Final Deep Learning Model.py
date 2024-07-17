#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries

# In[1]:


import os
import numpy as np
import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, GRU, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Downloading necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')


# # Load and Explore Data

# In[ ]:


# Load data
data = pd.read_json('/content/Sarcasm_Headlines_Dataset.json', lines=True)


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.isnull().sum()


# In[ ]:


data.is_sarcastic.value_counts()


# # Data Visualization

# In[ ]:


# Count the number of sarcastic and non-sarcastic headlines
label_counts = data['is_sarcastic'].value_counts()
labels = ['Non-Sarcastic', 'Sarcastic']

# Plot Bar Chart
plt.figure(figsize=(10, 5))
sns.barplot(x=labels, y=label_counts)
plt.title('Distribution of Sarcastic and Non-Sarcastic Headlines')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Plot Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(label_counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'salmon'])
plt.title('Distribution of Sarcastic and Non-Sarcastic Headlines')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# # Data Cleaning

# In[ ]:


# Checking for duplicate values
data['headline'].duplicated().sum()


# In[ ]:


# Drop duplicate headlines
data = data.drop(data[data['headline'].duplicated()].index, axis=0)


# In[ ]:


# Rechecking for duplicate values
data['headline'].duplicated().sum()


# In[ ]:


# Drop unnecessary columns
data = data.drop(columns=['article_link'])


# In[ ]:


data.head()


# # Text Preprocessing

# In[ ]:


# Define stopwords and punctuation to remove
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

# Define text preprocessing functions
def split_into_words(text):
    return text.split()

def to_lower_case(words):
    return [word.lower() for word in words]

def remove_punctuation(words):
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    return [re_punc.sub('', w) for w in words]

def keep_alphabetic(words):
    return [word for word in words if word.isalpha()]

def remove_stopwords(words):
    return [w for w in words if not w in stop]

def to_sentence(words):
    return ' '.join(words)

def denoise_text(text):
    words = split_into_words(text)
    words = to_lower_case(words)
    words = remove_punctuation(words)
    words = keep_alphabetic(words)
    words = remove_stopwords(words)
    return to_sentence(words)

# Apply text cleaning to the 'headline' column
data['news_headline'] = data['headline'].apply(denoise_text)


# In[ ]:


# Display the first few rows after text cleaning
data.head()


# # Splitting Data into Training and Testing Sets

# In[ ]:


# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data['news_headline'], data['is_sarcastic'], test_size=0.20, random_state=42
)

# Label encoding the target variable
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Display the first few rows of training data and labels
print(train_data.head())
print(train_labels_encoded[:5])


# # Tokenization and Padding

# In[ ]:


# Tokenization: converting text to sequences of integers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)   #updates internal vocabulary based on a list of texts
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

# Padding sequences to ensure uniform length
max_length = max([len(x) for x in train_sequences])
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

# Display the shape of the padded sequences
print(train_padded.shape)
print(test_padded.shape)


# # Building the CNN-LSTM Model

# In[ ]:


# Building the CNN-LSTM model
embedding_dim = 200  # Increased embedding dimension

cnn_lstm_model = Sequential()
cnn_lstm_model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))

# Adjusted convolutional layer parameters
cnn_lstm_model.add(Conv1D(1024, 12, activation='relu'))
cnn_lstm_model.add(MaxPooling1D(pool_size=4))
cnn_lstm_model.add(Dropout(0.3))

# Bidirectional LSTM layer with increased units
cnn_lstm_model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))
cnn_lstm_model.add(Dense(64, activation='relu'))
cnn_lstm_model.add(Dropout(0.3))

cnn_lstm_model.add(Dense(1, activation='sigmoid'))

# Compiling the model with a lower learning rate
cnn_lstm_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
print(cnn_lstm_model.summary())

# Adding EarlyStopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training the model
history_cnn_lstm = cnn_lstm_model.fit(
    train_padded, train_labels_encoded,
    epochs=5, batch_size=64, validation_data=(test_padded, test_labels_encoded),
    callbacks=[ReduceLROnPlateau(monitor='val_loss', patience=3, cooldown=0), early_stopping]
)

# Evaluate the model
cnn_lstm_raw_predictions = cnn_lstm_model.predict(test_padded)
cnn_lstm_predictions = np.where(cnn_lstm_raw_predictions > 0.5, 1, 0)

print("CNN-LSTM Model Accuracy:", accuracy_score(test_labels_encoded, cnn_lstm_predictions))


# In[ ]:


# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels_encoded, cnn_lstm_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
print("Classification Report:")
print(classification_report(test_labels_encoded, cnn_lstm_predictions))


# In[ ]:


# Plot heatmap for confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix - CNN-LSTM Model')
plt.show()


# # Making Predictions on New Data

# In[ ]:


# Function to preprocess, tokenize, and pad a new sentence
def preprocess_and_predict(sentence, tokenizer, model, max_length):
    def preprocess_text(text):
        # Denoise text using the same preprocessing steps
        stop = set(stopwords.words('english'))
        punctuation = list(string.punctuation)
        stop.update(punctuation)

        def split_into_words(text):
            return text.split()

        def to_lower_case(words):
            return [word.lower() for word in words]

        def remove_punctuation(words):
            re_punc = re.compile('[%s]' % re.escape(string.punctuation))
            return [re_punc.sub('', w) for w in words]

        def keep_alphabetic(words):
            return [word for word in words if word.isalpha()]

        def remove_stopwords(words):
            return [w for w in words if not w in stop]

        def to_sentence(words):
            return ' '.join(words)

        words = split_into_words(text)
        words = to_lower_case(words)
        words = remove_punctuation(words)
        words = keep_alphabetic(words)
        words = remove_stopwords(words)
        return to_sentence(words)

    # Preprocess the input sentence
    processed_sentence = preprocess_text(sentence)

    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([processed_sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Make prediction
    prediction = model.predict(padded_sequence)
    return "Sarcastic" if prediction > 0.5 else "Not Sarcastic"


# In[ ]:


# Example usage
new_sentence = "My name is Priyanshu."
result = preprocess_and_predict(new_sentence, tokenizer, cnn_lstm_model, max_length)
print(f'The sentence "{new_sentence}" is {result}.')


# In[ ]:


# Example usage
new_sentence = "I am busy right now, can I ignore you some other time?"
result = preprocess_and_predict(new_sentence, tokenizer, cnn_lstm_model, max_length)
print(f'The sentence "{new_sentence}" is {result}.')


# In[ ]:


# Example usage
new_sentence = "I love watching cricket"
result = preprocess_and_predict(new_sentence, tokenizer, cnn_lstm_model, max_length)
print(f'The sentence "{new_sentence}" is {result}.')


# In[ ]:


# Example usage
new_sentence = "I’m so thrilled to be working overtime on the weekend."
result = preprocess_and_predict(new_sentence, tokenizer, cnn_lstm_model, max_length)
print(f'The sentence "{new_sentence}" is {result}.')


# In[ ]:


# Example usage
new_sentence = "I love my boss who asks me to work overtime"
result = preprocess_and_predict(new_sentence, tokenizer, cnn_lstm_model, max_length)
print(f'The sentence "{new_sentence}" is {result}.')


# In[ ]:


# Example usage
new_sentence = "thirtysomething scientists unveil doomsday clock of hair loss"
result = preprocess_and_predict(new_sentence, tokenizer, cnn_lstm_model, max_length)
print(f'The sentence "{new_sentence}" is {result}.')


# In[ ]:


# Example usage
new_sentence = "this new orange era: the growing divide"
result = preprocess_and_predict(new_sentence, tokenizer, cnn_lstm_model, max_length)
print(f'The sentence "{new_sentence}" is {result}.')

