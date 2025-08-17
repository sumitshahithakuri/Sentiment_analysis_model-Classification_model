from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

data=pd.read_csv('./IMDB_Movie_reviews.csv')
print(data.info())
print("review: ",data["review"][49998],"sentiment",data["sentiment"][49998])