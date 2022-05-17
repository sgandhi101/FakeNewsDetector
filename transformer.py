import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submit = pd.read_csv('submit.csv')


# clean up data - fill NULL vals with string indicating no vals 
train = train.fillna("NULL data")
test = test.fillna("NULL data")

# our test labels are in the submit csv
labels = train['label'].values
texts = train['text'].values
testLabels = submit['label'].values
testText = test['text'].values


# this is our multi head attention layer
class MultiHeadAtt(layers.Layer):
    def __init__(self,num_heads,embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = num_heads
        self.head_dim = embed_dim//num_heads

        self.query = [layers.Dense(self.embed_dim,use_bias=False) for i in range(num_heads)]
        self.key = [layers.Dense(self.embed_dim,use_bias=False)for i in range(num_heads)]
        self.value = [layers.Dense(self.embed_dim,use_bias=False)for i in range(num_heads)]

        self.linOutput = layers.Dense(self.embed_dim,use_bias=False)


    def scaled_dot_product(self,query,key,value):
        #compute the attention
        #multiply the query matrix by the transposed key matrix
        numerator = tf.matmul(query,key,transpose_b=True)
        denominator = key.shape[1] ** 0.5
        #print(denominator)
        softmax = tf.nn.softmax(numerator/denominator)
        #print("\nSOFTMAX:  {0}, Type:  {1}".format(softmax,type(softmax)))
        return tf.matmul(softmax,value)


    def call(self,query,key,value):
        #all functions run through call in tensorflow, if we want to switch to pytorch, just rename the call functions 'forward'
        batchSize = tf.shape(query)[0]

        allHeads = []
        for head in range(self.heads):

            #get the individual heads and pass them through the linear layer
            q, k, v = self.query[head](query), self.key[head](key), self.value[head](value)

            #calculate the attention output
            output = self.scaled_dot_product(q,k,v)
            
            #combine all the output heads so we can concat them all after iterating through all individual heads
            allHeads.append(output)
            
        #now concat before we pass through the final linear layer
        concat = tf.concat(allHeads,axis = 2)
        #concat = tf.reshape(allHeads,(batchSize,allHeads[0].shape[1],self.embed_dim))

        #pass through the linearization layer
        output = self.linOutput(concat)

        return output


# the following is thanks to https://keras.io/examples/nlp/text_classification_with_transformer/
# this TransformerBlock class inherits from layers.Layer
# layers.Layer inherits from Module - which implements __call__, so this is why we can do
# an_object_of_this_class(x) in the later lines below
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.4):
        # call on the parent constructor=
        super(TransformerBlock, self).__init__()
 
        # this is our self-attention
        self.att = MultiHeadAtt(num_heads,embed_dim) 
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)


    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs, inputs) 
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


vocab_size = 20000
maxlen = 250
trainLabels = labels

train_size = len(texts)
test_size = len(testText)
tokenizer = keras.preprocessing.text.Tokenizer(num_words= vocab_size,split=' ',oov_token=0)
tokenizer.fit_on_texts(texts)

# the following lines is when we convert our data to sequences
trainSequence = tokenizer.texts_to_sequences([texts[i] for i in range(train_size)])
testSequence = tokenizer.texts_to_sequences([testText[i] for i in range(test_size)])

trainData = keras.preprocessing.sequence.pad_sequences(trainSequence, maxlen=maxlen)
testData = keras.preprocessing.sequence.pad_sequences(testSequence,maxlen=maxlen)

embed_dim = 40  # Embedding size for each token
num_heads = 3 # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
sg_model = keras.models.load_model('skipmodel')  # load in our trained model
# this is from our word2vec implementation
embedding_layer = sg_model.get_layer('w2v_embedding')
x = embedding_layer(inputs)

transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
# feel free to decrease this if your machine cannot handle this - the lower
# it gets sets to, the more time it takes
BATCH_SIZE = 1024

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/
"""
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = 5,
                                        restore_best_weights = True)
"""
# earlystopping, which calculates the best epoch to train on based off of training loss and validation loss, gave us an epoch number of 7
history = model.fit(
    trainData, trainLabels, batch_size=BATCH_SIZE, epochs=7, validation_data=(testData, testLabels))
model.save('transformermodel')  # this is to save our model into a dir called transformermodel

