import uvicorn
from fastapi import FastAPI
from smsdetection import SmsDetection
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger_eng")

app = FastAPI()
pickle_in = open("smsdetection.pkl", "rb")
classifier = pickle.load(pickle_in)
pickle_vector = open("vectorizer.pkl", "rb")
vector = pickle.load(pickle_vector)
# preprocessing functions
## converting text into lower
def convert_lower(text):
    return text.lower()

# removing punctuatios and special characters 
def remove_punctuations_specchars(text):
    result = re.sub(r'[^a-zA-Z\s]', '',text)
    return result


## removing stop words 

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


## lemmatization of the words 

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'a'
    elif tag.startswith("V"):
        return 'v'
    elif tag.startswith("N"):
        return 'n'
    elif tag.startswith("R"):
        return 'r'
    else:
        return 'n'

def lemmatizer(text):
    lemmatize = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    lematized_sentence = []
    for word,tag in tagged_tokens:
        if word.lower() == "are" or word.lower() in ["is", "am"]:
            lematized_sentence.append(word)
        else:
            lematized_sentence.append(lemmatize.lemmatize(word, get_wordnet_pos(tag)))
    return " ".join(lematized_sentence)

def preprocess_text(text):
    text = convert_lower(text)
    text = remove_punctuations_specchars(text)
    text = remove_stop_words(text)
    text = lemmatizer(text)
    return text

@app.get('/')
def index():
    return {'message': 'Hello, User'}

@app.post('/predict')
def predict_message(data:SmsDetection):
    data = data.dict()
    message = data["message"]
    text1 = preprocess_text(message)
    text = vector.transform([text1])
    print(classifier.predict(text))
    prediction = classifier.predict(text)
    if prediction[0] == 0:
        prediction = "Real"
    else:
        prediction = "Fake"
    return {
        'prediction': prediction
    }

if __name__ == "__main__":
    pass

