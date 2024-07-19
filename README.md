## Sarcasm Detection in News Headlines

This project uses a CNN-LSTM model to detect sarcasm in news headlines.

## Dataset

We use the Sarcasm Headlines Dataset containing labeled sarcastic and non-sarcastic headlines.
Link for dataset- https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

## Preprocessing

The text is cleaned by:
- Removing punctuation
- Converting to lowercase
- Removing stopwords
- Tokenizing and padding sequences

## Model

The model architecture includes:
- Embedding layer
- 1D Convolutional layer
- Max Pooling layer
- Bidirectional LSTM layer
- Dense layers for classification

## Training

The model is trained with:
- Binary cross-entropy loss
- Adam optimizer
- Early stopping and learning rate reduction to prevent overfitting

## Evaluation

The model's performance is evaluated using accuracy, confusion matrix, and classification report.


