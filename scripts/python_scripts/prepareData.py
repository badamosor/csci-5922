'''
Read dataset and create word dictionary for each language

reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''

from __future__ import unicode_literals, print_function, division
from io import open
import string
import re
import csv
import codecs
from tokenization import megasplit

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name, char_based):
        self.name = name
        self.char_based = char_based   #char_based = True or False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        
    def addSentence(self, sentence, char_based):
        if (char_based):

            tokenized_text = megasplit('', sentence)
            regex = "[\u0618-\u061A|\u064B-\u0653]"
            for i in range(len(tokenized_text)):
                if (tokenized_text[i] == ''):
                    continue
                if (re.search(regex, tokenized_text[i])):
                    continue
                if (re.search(regex, tokenized_text[i+1])):
                    char = tokenized_text[i] + tokenized_text[i+1]
                    self.addWord(char)
                else:
                    char = tokenized_text[i]
                    self.addWord(char)
        else:
            tokenized_text = megasplit(' ', sentence)
            for word in tokenized_text:
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(lang1, lang2, FILENAME_TRAIN, FILENAME_TEST, char_based):
    print("Reading lines...")   

    ENCODING = 'utf-8'
    train_pairs = []
    with codecs.open(FILENAME_TRAIN, "r") as fp:
      reader = csv.reader(fp)
      for rows in reader:
        train_pairs.append(rows)
        
    
    test_pairs = []
    with codecs.open(FILENAME_TEST, "r") as fp:
      reader = csv.reader(fp)
      for rows in reader:
        test_pairs.append(rows)
        
    input_lang = Lang(lang1, char_based)
    output_lang = Lang(lang2, char_based)
    
    return (input_lang, output_lang, train_pairs, test_pairs)

def prepareData(lang1, lang2, FILENAME_TRAIN, FILENAME_TEST, char_based):
    input_lang, output_lang, train_pairs, test_pairs = readLangs(lang1, lang2,FILENAME_TRAIN, FILENAME_TEST, char_based)
  
    print("Read %s training sentence pairs" % len(train_pairs))
  
    print("Counting words...")
    for pair in train_pairs:
        input_lang.addSentence(pair[0], char_based)
        output_lang.addSentence(pair[1], char_based)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    
    print("Read %s test sentence pairs" % len(test_pairs))
    print("Counting words...")
    for pair in test_pairs:
        #print("printing pairs:", pair[0], pair[1])
        input_lang.addSentence(pair[0], char_based)
        output_lang.addSentence(pair[1], char_based)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return (input_lang, output_lang, train_pairs, test_pairs)
