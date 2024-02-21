import nltk
import spacy
from nltk import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.chunk import ne_chunk
from pycontractions import Contractions
import re
import language_tool_python
from textblob import TextBlob

# text = input("enter the text : ")
nlp = spacy.load("en_core_web_sm")

# text = "Dedicated and detail-orient software developer for a passion to creating efficient, maintainable, and scalable software solutions."
text = input("enter the text : ")

tool = TextBlob(text)
print("Original Text: ", text)

print("#############################################################################################processing")

tool2 = language_tool_python.LanguageTool('en-US')

cont = Contractions(api_key="glove-twitter-100")
cont.load_models()

ex_text = list(cont.expand_texts([text]))[0]
print("Expanded text : ", ex_text)

cleaned = re.sub('[^A-Za-z0-9.,? ]', '', ex_text)
print("text after cleaning:: ", cleaned)

tokens = sent_tokenize(cleaned)
print("Token: ", tokens)

tags = pos_tag(tokens)
print("POS Tags: ", tags)

gr1 = tool.correct()
print("Text after corrected spellings: ", gr1)
print("\n")

gr2 = tool2.correct(text)
print("Text after correct grammer 1: ", gr2)
print("\n")

gr3 = tool2.correct(gr2)
print("Text after correct grammer 2: ", gr3)
print("\n")

gr4 = tool2.correct(gr3)
print("Text after correct grammer 3: ", gr4)
print("\n")


# final_gr1 = list(cont.contract_texts([gr1]))[0]
# # final_gr2 = list(cont.contract_texts([gr2]))[0]

# print("Suggestion 1 :: ", final_gr1)
# print("Suggestion 2 :: ", final_gr2)


# import re
# import nltk
# import string
# import language_tool_python
#
# punctuation = set(string.punctuation)
#
# # no_punct = "".join([word for word in text if word not in punctuation])
# # no_spec = re.sub('[^A-Za-z0-9 ]+','', no_punct)
# # lower_text = no_spec.lower()
# # tokens = word_tokenize(lower_text)
# # # clean_words = [word for word in words if word not in stop_words]
# # tagged_tokens = pos_tag(tokens)
# # ner_tags = ne_chunk(tagged_tokens)
#
# # preprocessed = " ".join()
#
# tool = language_tool_python.LanguageTool('en-US')
#
# text = "my name are shruti. my mother name are radha and he is a housewife"
#
# categories = ['TYPOS', 'GRAMMAR', 'STYLE', 'MISC', 'DICTIONARY']
# matches = tool.check(text, categories=categories)
#
# for error in matches:
#     print(text[error.offset: error.offset + error.errorLength])
#     if error.replacements:
#         print("Incorrect word: " + text[error.offset: error.offset + error.errorLength] + "in"+ 'context')
#         print("Replacement: " + error.replacements[0])
#         print("\n")
# print(tool.correct(text))
# tool.close()

        # def parsing(text):
        #     parsed_text = parser.parse(tokens)
        #     return parsed_text
        #
        #
        #
        # preprocessed_text = preprocess(text)
        # essay_tokens = tokenize(preprocessed_text)
        # for tokens in essay_tokens:
        #     tagged_tokens = tagging(tokens)
        #     print(tagged_tokens)
        #     parsed_text = parsing(tokens)
        #     for tree in parsed_text:
        #         print(tree)
        #     print("\n")

        # tokens = nltk.word_tokenize(text)
        # essay_tokens.append(tokens)
        # # print(tokens)
        # tagged_tokens = nltk.pos_tag(tokens)
        # print(tagged_tokens)
        # print("\n")


# def tokenize(text):
#     tokenized = sent_tokenize(text)
#     essay_tokens = []
#     for essay in tokenized:
#         tokens = word_tokenize(essay)
#         essay_tokens.append(tokens)
#     return essay_tokens
#
# def tagging(tokens):
#     tagged_tokens = pos_tag(tokens)
#     return tagged_tokens

# dir = "C:\\Users\\DeLL\\Desktop\\grammer\\essaytextfiles"
# os.chdir(dir)
#
# for filename in os.listdir(dir):
#     with open(filename, 'r', encoding='utf-8') as file:
#         text = file.read()


#
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
# Word2Vec, GloVe, or FastText,