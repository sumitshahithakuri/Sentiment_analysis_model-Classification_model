# importing required libraries and packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

# loading and previewing data set
data=pd.read_csv('./IMDB_Movie_reviews.csv')
# print(data.info())
# print("review: ",data["review"][49998],"sentiment",data["sentiment"][49998])

vectorizer=CountVectorizer()
v=vectorizer.fit_transform(data["review"])
# print(v.shape)

X_train,X_test,Y_train,Y_test=train_test_split(v,data["sentiment"],test_size=0.2,random_state=33)
# print(X_train.shape)

model=MultinomialNB()

model.fit(X_train,Y_train)

pred=model.predict(X_test)
accuracy=accuracy_score(Y_test,pred)
print("Accuracy: ",accuracy)

# lets test on my own review

my_review="""
I liked the movie but the character were not good as i expected and the music was bad at all.
"""

my_r_v=vectorizer.transform([my_review])
result=model.predict(my_r_v)
print(result[0])