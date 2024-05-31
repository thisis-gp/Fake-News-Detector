import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

fake = pd.read_csv("assets/Dataset/Fake.csv")
true = pd.read_csv("assets/Dataset/True.csv")

# creating a target column
fake['label'] = 1
true['label'] = 0

# data combination / concatenation
df = pd.concat([fake,true]).reset_index(drop=True)

# we only need text and label coloumn to train our model
df = df[['text','label']]

# Shuffle
from sklearn.utils import shuffle
df = shuffle(df).reset_index(drop=True)

from sklearn.model_selection import train_test_split
# Split the dataset
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the datasets to CSV files
train_data.to_csv('train1.csv', index=False)
test_data.to_csv('test1.csv', index=False)

new_df=pd.read_csv('train1.csv')

from nltk.stem import WordNetLemmatizer 
#creating instance
lemmatizer=WordNetLemmatizer()

import nltk
# Downloading Stopwords
nltk.download("stopwords")

# Obtaining Additional Stopwords From nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'ing'])

# import nltk
# import subprocess

# # Download and unzip WordNet
# try:
#     nltk.data.find('wordnet.zip')
# except:
#     nltk.download('wordnet', download_dir='/kaggle/working/')
#     command = "unzip /kaggle/working/corpora/wordnet.zip -d /kaggle/working/corpora"
#     subprocess.run(command.split())

# nltk.data.path.append('/kaggle/working/')

# # Now you can import the NLTK resources as usual
from nltk.corpus import wordnet

# Function to map POS tags to WordNet POS tags
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

from nltk.corpus import wordnet
# nltk.download('averaged_perceptron_tagger')
import spacy
# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text)
    result = []
    for token in doc:
        if token.text.lower() not in stop_words and token.lemma_ not in stop_words and len(token.text) > 2:
            pos = get_wordnet_pos(token.pos_)
            lemmatized_token = lemmatizer.lemmatize(token.text, pos)
            result.append(lemmatized_token)
    return ' '.join(result)
new_df['text'] = new_df['text'].astype(str)
new_df['text'] = new_df['text'].apply(preprocess)


from sklearn.feature_extraction.text import TfidfVectorizer
# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['text'])

print(tfidf_matrix.shape)  # Verify the shape of the TF-IDF matrix

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, new_df['label'], test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

def predict(user_input):
    dict = [{"text": user_input}]
    user_df = pd.DataFrame(dict)
    user_df['text'] = user_df['text'].apply(preprocess)
    new_tfidf_matrix = tfidf_vectorizer.transform(user_df['text'])
    new_y_pred = model.predict(new_tfidf_matrix)
    return new_y_pred

print(predict("Narendra modi is president of america"))
# print(predict("Former Secretary of State Hillary Clinton sought to arrange Pentagon and State Department consulting contracts for her daughter s friend, prompting concerns of federal ethics rules violations.Clinton in 2009 arranged meetings between Jacqueline Newmyer Deal, a friend of Chelsea Clinton and head of the defense consulting group Long Term Strategy Group, with Pentagon officials that involved contracting discussions, according to emails from Clinton s private server made public recently by the State Department. Clinton also tried to help Deal win a contract for consulting work with the State Department s director of policy planning, according to the emails.Deal is a close friend of Chelsea Clinton, who is vice chair of the Clinton Foundation. Emails between the two were included among the thousands recovered from a private email server used by the secretary of state between 2009 and 2013. Chelsea Clinton has described Deal as her best friend. Both Clintons attended Deal s 2011 wedding.Here s a little blurb from the fashion rag WWD on an event the two attended together: This story first appeared in the October 20, 2011 issue of WWD. Wearing a short-skirted black Chanel dress, Clinton began by crediting her longtime friend Jacqueline Newmyer.  Jackie invited me to see  Romeo and Juliet,  she said, remembering back to 1995.  The next time, I got my parents to come. And I have been coming here ever since.  That historic family night out occurred three years later. She and her parents arrived at the theater two days after Christmas 1998 and a week after the House of Representatives voted to impeach her father. The show they saw? None other than  Twelfth Night,  a tale of magical transformation. Talk about Freudian.Later, at the after party, Clinton elaborated on the nearly 20-year friendship with Newmyer, putting to lie the old Harry Truman quip  If you want a friend in Washington, get a dog.   Jackie and I are still best friends,  said Clinton, who met Newmyer her first year in Washington at the Sidwell Friends School.  She was in my wedding, and I was in hers.  Clinton, 31, continues to work on snagging her Oxford Ph.D. while working at New York University and with the Clinton Foundation and the Clinton Global Initiative. Newmyer, 32, is the president of the Long Term Strategy Group, a military research firm in Cambridge, Mass.Government cronyism, or the use of senior positions to help family friends, is not illegal. However, the practice appears to violate federal ethics rules that prohibit partiality, or creating the appearance of conflicts of interest.Specifically, the Code of Federal Ethics states that government employees  shall act impartially and not give preferential treatment to any private organization or individual.  Pentagon ethics guidelines also call for avoiding actions that would create even the appearance of improper behavior or conflicts of interest.The Clinton email exchanges with Deal between 2009 and 2011 were among tens of thousands of private emails made public by the State Department under pressure from Congress and the public interest law firm Judicial Watch.Read more: WFB"))




