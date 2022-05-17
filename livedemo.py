import pandas as pd
import tensorflow as tf
from tensorflow import keras
train = pd.read_csv('train.csv')
train = train.fillna("NULL data")
texts = train['text'].values

def makePrediction(input,model,tokenizer,vocab_size,maxlen):
    if input == "END":
        return -1
    else:
        #First we preprocess our input text then we tokenize
        #after we tokenize it, we pass it through our trained model to give us a prediction
        predSeq = tokenizer.texts_to_sequences([input])
        predData = keras.preprocessing.sequence.pad_sequences(predSeq,maxlen=maxlen)
        pred = model.predict(predData)
        if pred[0][0] >= 0.5:
            return "This a trustworthy text"
        else:
            return "This is Fake News"
def liveDemo(model,tokenizer,vocab_size,maxlen):
    inp = input("Enter a Text:  ")
    result = makePrediction(inp,model,tokenizer,vocab_size,maxlen)
    while result != -1:
        print(result)
        inp = input("Enter new text (END to stop):  ")
        result = makePrediction(inp,model,tokenizer,vocab_size,maxlen)

vocab_size = 20000
maxlen = 250
#load the transformer model
model = keras.models.load_model('transformermodel')

#set up our tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(num_words= vocab_size,split=' ',oov_token=0)
tokenizer.fit_on_texts(texts)

liveDemo(model,tokenizer,vocab_size,maxlen)
