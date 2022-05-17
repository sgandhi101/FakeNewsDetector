# credits to the following link for helping with the word embeddings:
# https://www.tensorflow.org/tutorials/text/word2vec
import io
import re
import string
import tensorflow as tf
import tqdm
import pandas as pd

from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from keras import callbacks

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# sentence = "The wide road shimmered in the hot sun" (this was a test sentence i used)


# tokenize : string -> [listof <string>]
# creates a list of words from a given sentence
def tokenize(sentence):
    return list(sentence.lower().split())


# make_vocab : [listof <string>] -> [dictof <string, int>]
# maps every unique string to a unique int in a dict
def make_vocab(tokens):
    words, ind = {}, 1
    words['<pad>'] = 0  # this is a padding token, we want to start from ind of 1
    for t in tokens:
        if t not in words:  # otherwise we will override a previous val in our dict
            words[t] = ind
            ind += 1
    return words


# inverse_vocab : [dictof <string, int>] -> [dictof <int, string>]
# reverses the types of the key, value pair in given dict
def inverse_vocab(words):
    return {ind: word for word, ind in words.items()}


# vectorize : [dictof <string, int>] -> [listof <int>]
# uses dict to convert words to its index mapping
def vectorize(tokens, words):
    return [words[w] for w in tokens]


# this is the num of neg samples per pos
# regarding this number, vals  between [5, 20] is shown to work best
# for smaller datasets, while num_ns between [2,5] suffices for larger datasets. 
num_ns = 3


# the above was an exercise i followed along with in order to understand
# what is going on in a smaller-scale

# gen_training_data : [listof <[listof <int>]>] int int int int -> [tupleof <[listof <Tensor>] [listof <Tensor>] [
# listof <Tensor>]>] generates skip-gram pairs with neg sampling
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  targets, contexts, labels = [], [], []

  # sampling table for tokens
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  for sequence in tqdm.tqdm(sequences):
    # positive skip-gram pairs for a sentence
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # produces pos context word & neg samples
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=SEED,
          name="negative_sampling")

      # context and label vectors (for one target word)
      negative_sampling_candidates = tf.expand_dims(
          negative_sampling_candidates, 1)
      context = tf.concat([context_class, negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # building the lists in the first line
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)
  return targets, contexts, labels


# this is where our texts are - generated the file with the code below
# texts is a vector we had indexed out from our original dataset
"""
test = pd.read_csv('test.csv')
rubbish = pd.read_csv('sg_texts/informal.csv')

test.fillna(" ")
rubbish.fillna(" ")
test = test['text'].values
rubbish = rubbish['Review'].values
test_ds = list(filter(lambda s: not str(s).isspace() and len(str(s)) > 100, test))
rubbish_ds = list(filter(lambda s: not str(s).isspace() and len(str(s)) > 100, rubbish))
f = open("sg_texts/ag_texts.txt", "w")
for test, diff in zip(test_ds, rubbish_ds):
    f.write(test + "\n")
    f.write(diff)
f.close()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
rubbish = pd.read_csv('sg_texts/informal.csv')

train.fillna(" ")
test.fillna(" ")
rubbish.fillna(" ")

train = train['text'].values
test = test['text'].values
rubbish = rubbish['Review'].values

test_ds = list(filter(lambda s: not str(s).isspace() and len(str(s)) > 100, test))
train_ds = list(filter(lambda s: not str(s).isspace() and len(str(s)) > 100, train))
rubbish_ds = list(filter(lambda s: not str(s).isspace() and len(str(s)) > 100, rubbish))

test_ds = test_ds[:len(test_ds)//2]
train_ds = train_ds[:len(train_ds)]

f = open("sg_texts/ag_texts.txt", "w")
for text, test, diff in zip(train_ds, test_ds, rubbish_ds):
    f.write(text + "\n")
    f.write(test + "\n")
    f.write(diff)
f.close()
"""

# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
path_to_file = 'sg_texts/kaggle_texts.txt' # our model was overfit when using word embeddings trained on the same data
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
test_ds = tf.data.TextLineDataset('sg_texts/ag_texts.txt').filter(lambda x: tf.cast(tf.strings.length(x), bool))


# custom_standardization : string -> string
# uses regex to lower strings and replace punctuation
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


vocab_size = 20000
sequence_length = 10
BATCH_SIZE = 1024

# here, we use the text vec layer to standardize strings and vectorize them (like we did up ^^^^ there)
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
vectorize_layer.adapt(text_ds.batch(BATCH_SIZE))

vectorize_test = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
vectorize_test.adapt(test_ds.batch(BATCH_SIZE))

# inverse mapping of dict
inverse_vocab = vectorize_layer.get_vocabulary()
inverse_test = vectorize_test.get_vocabulary()


# we are not vectorizing data from our texts.txt file
text_vector_ds = text_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=num_ns,
    vocab_size=vocab_size,
    seed=SEED)

test_vector_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE).map(vectorize_test).unbatch()
sequences = list(test_vector_ds.as_numpy_iterator())
ttargets, tcontexts, tlabels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=num_ns,
    vocab_size=vocab_size,
    seed=SEED)

# this is to configure our dataset for performance
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE) # cache() and prefetch() is what improves our performance... reminds me of memoization

validation = tf.data.Dataset.from_tensor_slices(((ttargets, tcontexts), tlabels))
validation = validation.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
validation = validation.cache().prefetch(buffer_size=AUTOTUNE) # cache() and prefetch() is what improves our performance... reminds me of memoization

# here comes the training and the creation of the model
# again, from the lovely guide linked above, our model will have the following layers:
"""
- target_embedding: A tf.keras.layers.Embedding layer which looks up the embedding of a word when it appears as a target word. The number of parameters in this layer are (vocab_size * embedding_dim).
- context_embedding: Another tf.keras.layers.Embedding layer which looks up the embedding of a word when it appears as a context word. The number of parameters in this layer are the same as those in target_embedding, i.e. (vocab_size * embedding_dim).
- dots: A tf.keras.layers.Dot layer that computes the dot product of target and context embeddings from a training pair.
- flatten: A tf.keras.layers.Flatten layer to flatten the results of dots layer into logits.
"""
class Word2Vec(Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=num_ns+1)
    self.dots = Dot(axes=(3, 2))
    self.flatten = Flatten()


  # this takes in a tuple of target, context and gets passed down to another embedding layer
  # (the only time we make a new embedding is when we initially take in our data; otherwise,
  # we use old outputs as a new input)
  def call(self, pair):
    target, context = pair
    word_emb = self.target_embedding(target)
    context_emb = self.context_embedding(context)
    dots = self.dots([context_emb, word_emb])
    return self.flatten(dots)


# compiling model - using keras loss calculation
# embedding_dim = 128
embedding_dim = 40
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy']) 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs") # so we can see the training stats
# https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/
"""
This stopped at epoch 6 - we will go on a bit more because this may be to do the weird validation data
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = 5,
                                        restore_best_weights = True)
"""
word2vec.fit(dataset, epochs=15, validation_data = validation, callbacks=[tensorboard_callback])
word2vec.save('skipmodel') # saving the trained model in a dir called skip model
