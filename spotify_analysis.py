import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob


spotify_stream_data = pd.read_csv("spotify_stream_data.csv", sep = ";")
print(spotify_stream_data.columns)

#------------------------------------ Removing Spotify Owned Playlists ----------------------------------#
df= spotify_stream_data[spotify_stream_data["owner"]!="spotify"]
print(df.sort_values("wau",ascending = False)[["wau","mau"]])
#removing the 95th percentile from data
df = df.query('mau < mau.quantile(.99)')
print(df.sort_values("wau",ascending = False)[["wau","mau"]])

#------------------------------------ Checking for Missing Values ---------------------------------------#
print(df.isnull().values.any())

#------------------------------------ Data Transformation and Creating Vbs  -----------------------------------------------#
# checking the data type for tokens field
print(type(df.tokens))
#converting list represnted in strings to array
def clean_tokens(x):
    y = x.strip('][').split(",")
    return(y)

def len_tokens(x):
    if x == "[]":
        return(0)
    else:
        return(len(x.strip('][').split(",")))

def clean_text(x):
    y = x.replace("-",None)
    return(y)

df["tokens_list"] = df["tokens"].apply(clean_tokens)
#checking for playlists with empty tokens of: len of array = 0
df["num_tokens"] = df["tokens"].apply(len_tokens)

#counting number of genres
df["hasgenre_1"] = df["genre_1"].apply(lambda z: 0 if z=='-' else 1)
df["hasgenre_2"] = df["genre_2"].apply(lambda z: 0 if z=='-' else 1)
df["hasgenre_3"] = df["genre_3"].apply(lambda z: 0 if z=='-' else 1)
df["num_genres"] = df["hasgenre_1"] + df["hasgenre_2"] +df["hasgenre_3"]

#counting number of moods
df["hasmood_1"] = df["mood_1"].apply(lambda z: 0 if z=='-' else 1)
df["hasmood_2"] = df["mood_1"].apply(lambda z: 0 if z=='-' else 1)
df["hasmood_3"] = df["mood_1"].apply(lambda z: 0 if z=='-' else 1)
df["num_moods"] = df["hasmood_1"] + df["hasmood_2"] +df["hasmood_3"]

df["genre_1"] = df["genre_1"].apply(lambda z: z.replace("-","NoLeadGenre"))
df["mood_1"] = df["mood_1"].apply(lambda z: z.replace("-","NoLeadMood"))


#--------------------------------------- Descriptive Statistics ------------------------------------------#
stream_data_no_outliers = df
print("Percentage of playlists with no tokens", stream_data_no_outliers[stream_data_no_outliers["num_tokens"] == 0].shape[0]*100/stream_data_no_outliers.shape[0])
print("Number of Playlists", stream_data_no_outliers["playlist_uri"].unique().shape[0])
print("Number of Owners", stream_data_no_outliers["owner"].unique().shape[0])

print("AVG Number of Track of Playlists", stream_data_no_outliers["n_tracks"].mean())
print("AVG Number of Albums of Playlists", stream_data_no_outliers["n_albums"].mean())
print("AVG Number of Artists of Playlists", stream_data_no_outliers["n_artists"].mean())
print("AVG Number of Tokens of Playlists", stream_data_no_outliers["num_tokens"].mean())
print("AVG Number of Streams of Playlists Today", stream_data_no_outliers["streams"].mean())
print("AVG Number of Streams30s of Playlists Today", stream_data_no_outliers["stream30s"].mean())
print("AVG Number of Streams30s of Playlists This Month",stream_data_no_outliers["monthly_stream30s"].mean())

print("AVG DAU", stream_data_no_outliers["dau"].mean())
print("MIN DAU", stream_data_no_outliers["dau"].min())
print("MAX DAU", stream_data_no_outliers["dau"].max())

print("AVG WAU", stream_data_no_outliers["wau"].mean())
print("MIN WAU", stream_data_no_outliers["wau"].min())
print("MAX WAU", stream_data_no_outliers["wau"].max())

print("AVG MAU",stream_data_no_outliers["mau"].mean())
print("MIN MAU", stream_data_no_outliers["mau"].min())
print("MAX MAU", stream_data_no_outliers["mau"].max())

#--------------------------------------- Exploratory Analysis  ------------------------------------------#
# Lead genres of Playlists
lead_genres = stream_data_no_outliers.groupby("genre_1")["playlist_uri"].count().reset_index().sort_values("playlist_uri",ascending = False)
objects = lead_genres.genre_1
y_pos = np.arange(len(objects))
genre_1 = lead_genres.playlist_uri

plt.bar(y_pos, genre_1, align='center', alpha=0.5, color = "darkgreen")
plt.xticks(y_pos, objects,rotation='vertical')
plt.title("Lead Playlist Genre Frequency")
plt.xlabel("Lead Playlist Genre")
plt.ylabel("Number of Playlists")
plt.tight_layout()
plt.show()

lead_moods = stream_data_no_outliers.groupby("mood_1")["playlist_uri"].count().reset_index().sort_values("playlist_uri",ascending = False)
objects = lead_moods.mood_1
y_pos = np.arange(len(objects))
mood_1 = lead_moods.playlist_uri
# histogram of Binomial distribution
plt.bar(y_pos, mood_1, align='center', alpha=0.5, color = "darkgreen")
plt.xticks(y_pos, objects,rotation='vertical')
plt.title("Lead Playlist Mood Frequency")
plt.xlabel("Lead Playlist Mood")
plt.ylabel("Number of Playlists")
plt.tight_layout()
plt.show()

wau_num_genres = df.sort_values("num_genres",ascending = True)
plt.scatter( wau_num_genres.num_genres,wau_num_genres.wau, color = "lime")
plt.title("Weekly Active Users to Number of Genres in Playlist")
plt.xlabel("Number Genres in Playlist")
plt.ylabel("WAU")
plt.tight_layout()
plt.show()

wau_num_tracks = stream_data_no_outliers.groupby("n_tracks")["wau"].mean().reset_index().sort_values("n_tracks",ascending = True)
plt.scatter( wau_num_tracks.n_tracks,wau_num_tracks.wau, color = "lime")
plt.title("AVG Weekly Active Users to Number of Tracks in Playlist")
plt.xlabel("Number Tracks in Playlist")
plt.ylabel("AVG WAU")
plt.tight_layout()
plt.show()

wau_num_albums = stream_data_no_outliers.groupby("n_albums")["wau"].mean().reset_index().sort_values("n_albums",ascending = True)
plt.scatter( wau_num_albums.n_albums,wau_num_albums.wau, color = "lime")
plt.title("AVG Weekly Active Users to Number of Albums in Playlist")
plt.xlabel("Number Albums in Playlist")
plt.ylabel("AVG WAU")
plt.tight_layout()
plt.show()

wau_num_artists = stream_data_no_outliers.groupby("n_artists")["wau"].mean().reset_index().sort_values("n_artists",ascending = True)
plt.scatter( wau_num_artists.n_artists,wau_num_artists.wau, color = "lime")
plt.title("AVG Weekly Active Users to Number of Artists in Playlist")
plt.xlabel("Number Artists in Playlist")
plt.ylabel("AVG WAU")
plt.tight_layout()
plt.show()

wau_num_tokens = df.groupby("num_tokens")["wau"].mean().reset_index().sort_values("num_tokens",ascending = True)
objects = wau_num_tokens.num_tokens
y_pos = np.arange(len(objects))
ntokens = wau_num_tokens.wau
plt.bar(y_pos, ntokens, align='center', alpha=0.5, color = "darkgreen")
plt.xticks(y_pos, objects)
plt.title("AVG Weekly Active Users to Number of Tokens of Playlist")
plt.xlabel("Number Tokens in Playlist")
plt.ylabel("AVG WAU")
plt.tight_layout()
plt.show()
#--------------------------------------- Sentiment Analysis  ------------------------------------------#
def get_sentiment_score(text):
    blob = TextBlob(text.strip("]["))
    blob = blob.replace('"', '')
    sentiment=blob.sentiment.polarity
    return(sentiment)


df["sentiment_score_tokens"] = df["tokens"].apply(get_sentiment_score)
print(df.sort_values("sentiment_score_tokens", ascending = False)[["tokens","sentiment_score_tokens"]].head(20))
print(df.sort_values("sentiment_score_tokens", ascending = True)[["tokens","sentiment_score_tokens"]].head(100))


wau_sentimentscore = df.sort_values("sentiment_score_tokens", ascending = True)
plt.scatter( wau_sentimentscore.sentiment_score_tokens,wau_sentimentscore.wau, color = "lime")
plt.title("Weekly Active Users to Sentiment Score of Playlist Tokens")
plt.xlabel("Tokens Sentiment Score")
plt.ylabel("WAU")
plt.tight_layout()
plt.show()


#--------------------------------------- Linear Regression  ------------------------------------------#
# create proxy for owners engagement: dummy
# owner_numplaylist = df["owner"].value_counts().to_dict()
# owner_numpl = []
# for k in range(len(df.owner)):
#     owner_id = df.owner[k]
#     num_pl = owner_numplaylist.values()[0].keys()[owner_id]
#     owner_numpl.append(num_pl)
# print(owner_numpl)
# pd.merge()
#df["owner_num_playlists"] = owner_numplaylists


# create dummies for the most popular genres and moods and adding to the data, the desired genres and moods
genres_moods = pd.get_dummies(df[['genre_1','mood_1']])
genres_moods = genres_moods[["genre_1_Indie Rock","genre_1_Rap", "genre_1_Pop", "mood_1_Defiant", "mood_1_Excited"]]
data = pd.concat([df,genres_moods], axis = 1)
df_sample = data.sample(15975).reset_index()

features = df_sample[["genre_1_Indie Rock","genre_1_Rap", "genre_1_Pop", "mood_1_Defiant", "mood_1_Excited",
                      "stream30s", "n_tracks", "n_local_tracks", "n_artists", "n_albums","num_genres",
                      "num_tokens","sentiment_score_tokens","users"
                      ]]
print(features)
success_metric = df_sample.wau
print(success_metric)

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
model = sm.OLS(success_metric, features)
fit_model = model.fit()
print(fit_model.summary())


from textblob import TextBlob


def get_sentiment_score(text):
    blob = TextBlob(text.strip("]["))
    blob = blob.replace('"', '')
    sentiment = blob.sentiment.polarity
    return (sentiment)


print("sentiment score is:", get_sentiment_score("[Today, the whether is great]"))



