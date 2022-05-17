# Fake News Detection

<p align="center">
<i>In light of recent events, the ability to detect fake news has proven itself to be important. By utilizing different algorithms, we are able to determine what is considered fake news - news that has been proven as factually incorrect - to certain degrees of accuracy.

We explore how two main types of algorithms (Naive Bayes and a Transformer Model) can be used and which one is more beneficial to accomplish this task.
</i>
</p>

## Base Requirements

Below are specific instructions on how to run our programs, along with dependencies used in the project.

Both programs will require **Python v3.8.6**.

In addition, you will need pip:

### Installing pip on Ubuntu

```
$ sudo apt-get install python3-pip
```

### Installing pip on Windows

- Download PIP get-pip.py
- Launch Windows Command Prompt
- Navigate to the location of get-pip.py and type the following:

```
$ python3 get-pip.py
```

* Verify installation by:

```
$ pip help
```

## NAIVE BAYES
In order to run the naive bayes model, ensure you are in the same directory as `naive_bayes.py` and type the following command in your terminal:

```
$ python3 naive_bayes.py
```

### Dependencies
```
$ pip install pandas

$ pip install rich

$ pip install nltk
```

(psst. you can also just run `$ pip install requirements.txt` for everything)

💡 On first run, [nltk](https://www.nltk.org) will ask you to install a specific corpus. I wasn't able to find this to remove it from my machine and make these instructions, but I know my groupmates were able to follow the very specific instructions nltk provides and install the requisite data.

## TRANSFORMER
In order to run the program that implements a transformer model, make sure you are in the same directory as `transformer.py` and type the following command in your terminal:

```
$ python3 transformer.py
```

This will train our model with our saved and trained word embedding. In order to interact with the model, you will have to run:

```
$ python3 livedemo.py
```

where you can input your own text and see it classified on the spot.


If you are interested in training our word embeddings again, run:

```
$ python3 embedding.py
```


### Dependencies
You will need the following commands to resolve any dependency issues:

```
$ pip install pandas

$ pip install tensorflow
```
