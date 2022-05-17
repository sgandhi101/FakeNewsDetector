import pandas as pd
import math
from nltk.tokenize import word_tokenize
from rich.progress import track

stopwords = list(pd.read_csv("stopwords.lst").iloc[:, 0])


class NaiveBayes:
    class Class:
        def __init__(self, x, N):
            """Instantiates a Class object.

            Args:
                x (pandas.DataFrame): The training data DataFrame.
                N (integer): The total number of values in the training dataset.
            """
            self.prior = len(x) / N
            self.likelihoods = {}

            def build_frequency_map(words):
                """Builds a frequency table for a list of words.

                Args:
                    words (List[str]): A list of words, in our case output from the nltk tokenizer.

                Returns:
                    dict: A frequency mapping, with words as keys and their frequency as values.
                """
                freq = {}
                # track is from the rich module, which creates a progress bar for this loop
                for word in track(words, description="Building frequency map..."):
                    if word in freq:
                        freq[word] += 1
                    else:
                        freq[word] = 1
                return freq

            self.words = []
            for v in track(x, description="Tagging train data..."):
                print("")
                # word_tokenize is an nltk function that intelligently splits (tokenizes) our corpus
                self.words += word_tokenize(v)

            self.word_frequency = build_frequency_map(self.words)
            self.unique_words = self.word_frequency.keys()

        def conditional(self, word):
            """Return a conditional probability for a given word.

            Args:
                word (string): A single word, hopefully in our trained set but not necessarily.

            Returns:
                float: A conditional probability for the word's incidence in *this particular* class.
            """
            if word not in self.likelihoods:
                self.likelihoods[word] = (
                    (1 + self.word_frequency[word])
                    if word in self.word_frequency
                    else 1
                ) / (len(self.unique_words) + len(self.words))
            return self.likelihoods[word]

        def posterior(self, x: str):
            """Returns a posterior possibility representing the likelihood a string x is a member of *this* class.

            Args:
                x (str): a string, presumably of real or fake news

            Returns:
                float: the posterior probability of this string being a part of this class
            """
            words = x.split()
            if len(words):  # if the message is empty, don't try and take log(0)
                return math.log(self.prior) + sum(
                    math.log(self.conditional(word)) for word in words
                )
            else:
                return math.log(self.prior)

    def __init__(self):
        self.classes = {}

    def train(self, x, y):
        """Trains a Naive Bayes classifier for given x and y data

        Args:
            x (pandas.Series): [description]
            y (pandas.Series): [description]
        """
        for _class in y.unique():
            self.classes[_class] = NaiveBayes.Class(x[y == _class], len(x))

    def predict(self, x: str):
        """Return a class prediction for a given string.

        Args:
            x (str): A string to be predicted.

        Returns:
            int: A predicted class, based on x.
        """
        class_probabilities = {
            label: _class.posterior(x) for label, _class in self.classes.items()
        }
        return max(class_probabilities, key=lambda x: class_probabilities[x])

    def accuracy(self, x, y):
        """Return an accuracy metric computed by running NaiveBayes.predict a bunch of times.

        Args:
            x (pandas.Series): strings of articles, tweets, and the like.
            y (pandas.Series): correct labels for the values in x.

        Returns:
            float: Accuracy of predicted data (correct/total).
        """
        assert len(x) == len(y), "x and y size does not match"

        predictions = []
        for v in track(x, description="Testing..."):
            predictions += [self.predict(v)]
        correct = predictions == y

        return correct.value_counts() / len(x)

    def confusion_matrix(self, x, y):
        assert len(x) == len(y), "x and y size does not match"

        predictions = []
        for v in track(x, description="Testing..."):
            predictions += [self.predict(v)]

        conf_matrix = [[0 for _ in self.classes] for _ in self.classes]

        for p_i, y_i in zip(predictions, y):
            conf_matrix[y_i][p_i] += 1

        return conf_matrix


if __name__ in "__main__":
    df = pd.read_csv("train.csv")
    df = df.astype({"id": int, "title": str, "author": str, "text": str, "label": int})

    # df["text"] = df["text"].str.replace(r"[^\w\s]+", "", regex=True)

    # df["text"] = df["text"].str.lower()

    # df["text"] = df["text"].str.replace(
    #    "\\b(" + "|".join(stopwords) + ")\\W", "", regex=True
    # )

    border = 3 * (len(df) // 4)
    train_df, test_df = df[:border], df[border:]

    n = NaiveBayes()
    n.train(train_df["text"], train_df["label"])
    print(n.confusion_matrix(test_df["text"], test_df["label"]))

    while True:
        inp = input(">> ")
        print(("Real.", "Fake!")[n.predict(inp)])
