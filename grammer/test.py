import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('./data/resume_dataset/withskill21.csv')
sim = []

tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(df['Resume'])
features = tfidf.get_feature_names_out()
values = matrix.toarray()
print(values)

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectorizer.fit(df['Resume'])
# resume_tfidf = tfidf_vectorizer.transform(df['Resume'])
# resume_tfidf_array = resume_tfidf.toarray()
# numeric_df = pd.DataFrame(resume_tfidf_array, columns=tfidf_vectorizer.get_feature_names_out())
# df_numeric = pd.concat([df, numeric_df], axis=1)
# print(df_numeric.head())

#
# for i in range (int(df['Resume'].iloc[-1])):
#     for j in range(i+1, (df['Resume'].iloc[-1])):
#         if (df['Resume'].iloc[i] == df['Resume'].iloc[j]):
#             sim.extend(['j'])
#     print(f'group[{i}] = {sim}')
############################################################################################################################################


# import pandas as pd
# import re
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
#
# dataset = pd.read_csv('./data/resume_dataset/UpdatedResumeDataSet.csv')
#
# resumes = dataset["Resume"]
# categories = dataset["Category"].unique()
#
# def extract(resume):
#     skills = re.findall(r"\b\w+\b", resume)
#     return skills
#
# cat_skills = {}
# for category in categories:
#     category_resumes = resumes[dataset["Category"] == category]
#     cat_skills[category] = set()
#
#     for resume in category_resumes:
#         skills = extract(resume)
#         cat_skills[category].update(skills)
#
# for category, skills in cat_skills.items():
#     print("Category: ", category)
#     print("Skills", ", ".join(skills))
#     print()
# extracted = [extract(resume) for resume in resumes]
# print(skills)


####################################################################################################################


# import nltk
# from nltk import word_tokenize
# from nltk import pos_tag
# from nltk.chunk import ne_chunk
# from pycontractions import Contractions
# import re
# import language_tool_python
# from gingerit.gingerit import GingerIt
#
# tool = GingerIt()
# tool2 = language_tool_python.LanguageTool('en-US')
# text = "i'll yours in a few days "
#
# cont = Contractions(api_key="glove-twitter-25")
# cont.load_models()
#
# ex_text = str(list(cont.expand_texts([text])))
#
# cleaned = []
# no_spec = re.sub('[^A-Za-z0-9.,?]+ ', '', ex_text)
# cleaned = "".join(no_spec)
# print("text after cleaning and expanding :: ",cleaned)
#
# tokens = word_tokenize(cleaned)
# tagged = pos_tag(tokens)
# ner_tags = ne_chunk(tagged)
#
# gr1 = tool.parse(cleaned)['result']
# gr2 = tool2.correct(gr1)
#
#
# print("correct grammar :: ",gr1)
# print("gr2:: ", gr2)
#
#
# ______________________________________________________________________________________________________________________________________________
#

