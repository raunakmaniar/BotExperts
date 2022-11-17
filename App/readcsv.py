from transformers import BertTokenizer, BertForSequenceClassification,pipeline,AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# text='Some businesses, however, have been accused by activists of "greenwashing", with splashy announcements of programmes that do little if anything to reduce overall greenhouse gas emissions'
# score=esgScore(text)
# print(score)#[{'label': 'Environmental', 'score': 0.9917951226234436}]
import spacy
import os
from spacy import displacy
from collections import Counter
import en_core_web_sm
import pandas as pd
import csv
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
nlp = en_core_web_sm.load()

def prediction_classifier(sentence):
    finBert=BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg',num_labels=4)
    tokeniZer=BertTokenizer.from_pretrained('yiyanghkust/finbert-esg')
    esgScore=pipeline("text-classification",model=finBert,tokenizer=tokeniZer)
    return esgScore(sentence)

def remove_mystopwords(sentence):
    text_tokens = word_tokenize(sentence)
    tokens_filtered= [word for word in text_tokens if not word in my_stopwords]
    return (" ").join(tokens_filtered)

def text_classifier(sentence):
  tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
  model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
  classifier=pipeline(task="text-classification",model=model,tokenizer=tokenizer)
  return classifier(sentence)

def read_csv():
    curr_path = os.getcwd()
    file_path = os.path.join(curr_path,'dataset','testData.csv')
    return file_path

def process_new(new):
    try:
        exclude = set(string.punctuation)
        nt = re.sub(r"[\n\t\s]*", "", new)
        st = ''.join(ch for ch in new if ch not in exclude)
        score_esg=prediction_classifier(nt)
        rating = text_classifier(nt)
        print(score_esg,rating)
        doc = nlp(st)
        org_cnt = {}
        for X in doc.ents:
            if X.label_ == 'ORG' :
                if X.text in org_cnt:
                    org_cnt[X.text] = org_cnt[X.text] + 1
                else:
                    org_cnt[X.text] = 1
            if X.label_ == 'MONEY':
                print("Mony",X.text)
        print(org_cnt)
        return {'News':new,'ESG':score_esg,'Score':rating}
    except Exception as e:
        print("exception" ,e)
        pass


if __name__ == '__main__':
    data = pd.read_csv(read_csv(),usecols = ['News'],encoding="utf-8")
    with open('dumpnew-2.csv', 'w', newline='') as file:
        fieldnames = ['News', 'ESG','Score']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        for i, row in data.iterrows():
            for j, col in row.iteritems():
                res = process_new(col)
                print("res",res)
                if res != None:
                    writer.writerow(res)

