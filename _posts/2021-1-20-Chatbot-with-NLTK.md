---
layout: post
title: Building a Simple Chatbot with Python using NLTK
---


### Introduction
 A chatbot is an artificial intelligence-powered piece of software in a device, application, website or other networks that try to gauge consumer’s needs and then assist them to perform a particular task like a commercial transaction, hotel booking, form submission etc, the possibilities are almost endless.
Chatbots are extremely helpful for business organizations and also the customers. Facebook released data that proved the value of bots, more than 2 billion messages are sent between people and companies monthly. The HubSpot research tells us that 71% of people want to get customer support from messaging apps. It is a quick way to get their problems solved so chatbots have a bright future in organizations. And today almost every company has a chatbot deployed to engage with the users.

### How do Chatbots work?
Chatbots are interestingly nothing but an intelligent piece of software that can interact and communicate with people just like humans. And there are two main variants of  [chatbots:](https://medium.com/botsupply/rule-based-bots-vs-ai-bots-b60cdb786ffa)  **Rule-Based** and **Self-learning**

> In this article, we are going to build a simple project on Chatbot by using NLTK library in python

### Building the Bot

#### Pre-requisites
To implement the chatbot, we will be using **scikit-learn** library and **NLTK** which is a Natural Language Processing toolkit. However, if you are new to NLP, you can read the article and then refer back to resources.
#### NLP
NLP is the field of study that focuses on the interactions between human language and computers. It is at the intersection of computer science, artificial intelligence, and computational linguistics[Wikipedia]. All chatbots come under the NLP (Natural Language Processing) concepts also.
**NLTK(Natural Language Toolkit)** is a leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources such as WordNet, along with a suite of text processing libraries for classification, tokenization, stemming, tagging, parsing, and semantic reasoning, wrappers for industrial-strength NLP libraries.
#### Downloading and installing NLTK

1. Install NLTK: run 
```pip install nltk
``` 
2. Importing the necessary libraries
```
       import nltk
       import numpy as np
       import random
       import string # to process standard python strings
``` 
#### Reading in and pre-processing the data with NLTK
We will be using the Wikipedia page for [chatbots](https://en.wikipedia.org/wiki/Chatbot) as our data collection. Copy the contents from the page and place it in a text file named ‘chatbot.txt’. However, you can use any corpus of your choice.
The main issue with text data is that it is all in text format (strings). However, Machine learning algorithms need some sort of numerical feature vector in order to perform the task. So it has to go through a lot of pre-processing for the machine to easily understand. For textual data, there are many preprocessing techniques available. The first technique is tokenizing, in which we break the sentences into words. We will read in the chatbot.txt file and convert the entire text collection data into a list of sentences and a list of words for further pre-processing. Basic text pre-processing includes:
  - Converting the entire text into uppercase or lowercase, so that the algorithm does not treat the same words in different cases as different
  - **Tokenization**: Tokenization is just the term used to describe the process of converting the normal text strings into a list of tokens i.e words that we actually want. Sentence tokenizer can be used to find the list of sentences and Word tokenizer can be used to find the list of words in strings.

The NLTK data package includes a pre-trained Punkt tokenizer for English. We will use *nltk.download()* to chose and download the models and resources we needed. 

```
     f=open('chatbot.txt','r',errors = 'ignore')
     raw=f.read()
     raw=raw.lower()# converts to lowercase
     nltk.download('punkt') # first-time use only
     nltk.download('wordnet') # first-time use only
     sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
     word_tokens = nltk.word_tokenize(raw)# converts to list of words
``` 
**Pre-processing the raw text using Lemmatization**

We can convert words into the lemma form so that we can reduce all the canonical words. For example, the words play, playing, plays, played, etc. will all be replaced with play. This way, we can reduce the number of total words in our vocabulary. We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.

```
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
``` 
**Keyword matching**
We shall define a function for a greeting by the bot i.e if a user’s input is a greeting, the bot shall return a greeting response.

```
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
``` 
**Generating Response**
To generate a response from our bot for input questions, the concept of document similarity will be used. We begin by importing the modules from scikit-learn.
 - From scikit learn library, import the  [TFidf vectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) to convert a collection of raw documents to a matrix of TF-IDF features.

```
from sklearn.feature_extraction.text import TfidfVectorizer
``` 
 - Also, import  [cosine similarity](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html) module from scikit learn library

```
from sklearn.metrics.pairwise import cosine_similarity
``` 

This will be used to find the similarity between words entered by the user and the words in the collection data. 

We will define a function to response which searches the user’s utterance for one or more known keywords and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, it returns a response:” I am sorry! I don’t understand you”

```
def response(user_response):
    alice_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        alice_response=alice_response+"I am sorry! I don't understand you"
        return alice_response
    else:
        alice_response = alice_response+sent_tokens[idx]
        return alice_response
``` 
Finally, we will feed the lines that we want our bot to say while starting and ending a conversation depending upon the user’s input.

```
flag=True
print("ALICE: My name is Alice. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ALICE: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("ALICE: "+greeting(user_response))
            else:
                print("ALICE: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("ALICE: Bye! take care..")
``` 
We have coded our first chatbot in NLTK. Even though the chatbot couldn’t give a satisfactory answer for some questions, it fared pretty well on others.











