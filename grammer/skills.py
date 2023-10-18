import re
import pandas as pd
import nltk
import string
from resume_parser import resumeparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


nltk.download('punkt')
nltk.download('stopwords')

dataset = pd.read_csv('C:\\Users\\DeLL\\Desktop\\datasets for resume\\UpdatedResumeDataSet.csv')

parser = resume_parser.ResumeParser()

stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# with open('C:\\Users\\DeLL\\Desktop\\datasets for resume\\UpdatedResumeDataSet.csv', 'r', encoding='utf-8') as file:
#     text = file.read()

def preprocess(text):
    no_punct = "".join([word for word in text if word not in punctuation])

    no_spec = re.sub('[^A-Za-z0-9 ]+','', no_punct)

    lower_text = no_spec.lower()

    words = word_tokenize(lower_text)

    clean_words = [word for word in words if word not in stop_words]

    preprocessed = " ".join(clean_words)
    return preprocessed

for index, row in dataset.iterrows():
    resume_text = row['Resume']
    preprocessed_text = preprocess(resume_text)
    parsed_resume = parser.parse(preprocessed_text)

#
# def tokenize(text):
#     tokenized = sent_tokenize(text)
#     essay_tokens = []
#     for essay in tokenized:
#         tokens = word_tokenize(essay)
#         essay_tokens.append(tokens)
#     return essay_tokens
#
#
# def tagging(tokens):
#     tagged_tokens = pos_tag(tokens)
#     return tagged_tokens
#
#
# preprocessed_text = preprocess(text)
# res_tokens = tokenize(preprocessed_text)
# for tokens in res_tokens:
#     tagged_tokens = tagging(tokens)
#     print(tagged_tokens)
#     print("\n")

