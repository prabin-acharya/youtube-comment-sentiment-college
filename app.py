from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

app = Flask(__name__)

def clean_text(data):

    # remove hashtags and @usernames
    data = re.sub(r'@[A-Za-zA-Z0-9]+', '', data)  # removing @mentions
    data = re.sub(r'@[A-Za-z]+', '', data)        # removing @mentions
    data = re.sub(r'@[-)]+', '', data)            # removing @mentions
    data = re.sub(r'#', '', data )                # removing '#' sign
    data = re.sub(r'RT[\s]+', '', data)           # removing RT
    data = re.sub(r'https?\/\/\S+', '', data)     # removing the hyper link
    data = re.sub(r'&[a-z;]+', '', data)          # removing '>'

    # tekenization using nltk
    data = word_tokenize(data)

    return data

# ====
# Number of labels: joy, anger, fear, sadness, neutral
num_classes = 5

# Number of dimensions for word embedding
embed_num_dims = 300

# Max input length (max number of words)
max_seq_len = 500

class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']

# 

data_train = pd.read_csv('data_train.csv', encoding='utf-8')
data_test = pd.read_csv('data_test.csv', encoding='utf-8')

X_train = data_train.Text
X_test = data_test.Text

y_train = data_train.Emotion
y_test = data_test.Emotion

# data = data_train.append(data_test, ignore_index=True)
data = pd.concat([data_train, data_test], ignore_index=True)


# 

texts = [' '.join(clean_text(text)) for text in data.Text]

texts_train = [' '.join(clean_text(text)) for text in X_train]
texts_test = [' '.join(clean_text(text)) for text in X_test]

# 

#Tokenize

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequence_train = tokenizer.texts_to_sequences(texts_train)
sequence_test = tokenizer.texts_to_sequences(texts_test)

index_of_words = tokenizer.word_index

# vocab size is number of unique words + reserved 0 index for padding
vocab_size = len(index_of_words) + 1

print('Number of unique words: {}'.format(len(index_of_words)))

# 

#Padding

X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len )
X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len )


# 


encoding = {
    'joy': 0,
    'fear': 1,
    'anger': 2,
    'sadness': 3,
    'neutral': 4
}

# Integer labels
y_train = [encoding[x] for x in data_train.Emotion]
y_test = [encoding[x] for x in data_test.Emotion]

# 



# =====

# Load the trained model
model_path = 'EmotionModel.h5'  # Update with your actual model path
loaded_model = load_model(model_path)

# Tokenizer setup (you can reuse this from your training code)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

# Max sequence length
max_seq_len = 500

# @app.route('/predict_emotion', methods=['POST'])
# def predict_emotion():
#     data = request.get_json(force=True)
#     message = data['message']

#     # Tokenize and pad the input message
#     seq = tokenizer.texts_to_sequences([message])
#     padded = pad_sequences(seq, maxlen=max_seq_len)

#     # Make predictions
#     pred = loaded_model.predict(padded)

#     # Convert predictions to class names
#     class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
#     predicted_class = class_names[np.argmax(pred)]

#     return jsonify({'message': message, 'predicted_emotion': predicted_class})


@app.route('/predict_emotion', methods=['GET', 'POST'])
def predict_emotion():
    if request.method == 'GET':
        return render_template('predict_emotion.html')
    elif request.method == 'POST':
        message = request.form['message']

        # Tokenize and pad the input message
        seq = tokenizer.texts_to_sequences([message])
        padded = pad_sequences(seq, maxlen=max_seq_len)

        # Make predictions
        pred = loaded_model.predict(padded)

        # Convert predictions to class names
        class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
        predicted_class = class_names[np.argmax(pred)]

        return jsonify({'message': message, 'predicted_emotion': predicted_class})

if __name__ == '__main__':
    app.run(port=5000)
