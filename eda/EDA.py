import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from utils import data_loader as dl


# Streamlit dependencies
import streamlit as st


# Load data
# Load data
train_df = dl.load_dataframe(
    '../unsupervised_data/unsupervised_movie_data/train.csv', index=None)
imdb_df = dl.load_dataframe(
    '../unsupervised_data/unsupervised_movie_data/imdb_data.csv', index=None)
movies_df = dl.load_dataframe(
    '../unsupervised_data/unsupervised_movie_data/movies.csv', index='movieId')


decades = [(1870, 1879), (1880, 1889), (1990, 1909), (1910, 1919), (1920, 1929),
           (1930, 1939), (1940, 1949), (1950, 1959), (1960, 1969), (1970, 1979),
           (1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2022), (9999, 10000)]

decade_categories = ['1870s', '1880s', '1890s', '1900s', '1910s', '1920s', '1930s', '1940',
                     '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s', 'Unspecified']


def get_release_dates(title):

    sub = title[-5:-1]
    year = int(sub) if sub.isdigit() else 9999
    return year


def get_releases_by_year(df, release_years):

    mask = df[(df['release_year'] >= release_years[0]) &
              (df['release_year'] <= release_years[-1])]
    return [mask[mask['release_year'] == year]['movieId'].nunique() for year in release_years]


def count_ratings_by_years(df, start, end):

    ratings_count = [0] * 10
    ratings = np.linspace(0.5, 5.0, 10)
    for year in range(start, end + 1):
        df_year = df[df['rating_year'] == year]
        count = 0
        for rating in ratings:
            ratings_count[count] += (df_year[df_year['rating']
                                     == rating]['movieId'].count())
            count += 1
    return ratings_count


def get_genre_count(number_of_genres, movie_genres, df):

    genre_count = [0] * len(movie_genres)
    for index, genres in df[df['genre_count'] == number_of_genres]['genres'].items():
        for genre in genres.split('|'):
            genre_count[movie_genres.index(genre)] += 1

    return genre_count


def movie_rating_decade_released(start_year, end_year, decades, df):

    ratings_count = []
    ratings_average = []

    for start, end in decades:
        mask_1 = (df['release_year'] >= start) & (df['release_year'] <= end)
        mask_2 = (df['rating_year'] >= start_year) & (
            df['rating_year'] <= end_year)
        sub_df = df[mask_1 & mask_2]['rating']
        ratings_count.append(sub_df.count())
        ratings_average.append(np.round(sub_df.mean(), 2))

    return ratings_count, ratings_average


def get_user_ids(year, user_ids, df):

    users = []
    for user_id in user_ids:
        if df[df['userId'] == user_id]['rating_year'].max() <= year:
            users.append(user_id)

    return users


def feature_frequency(df, column):
    """
    Function to count the number of occurences of metadata such as genre
    Parameters
    ----------
        df (DataFrame): input DataFrame containing movie metadata
        column (str): target column to extract features from
    Returns
    -------

    """
    # Creat a dict to store values and drop nan values
    df = df.dropna(axis=0)
    genre_dict = {f'{column}': list(),
                  'count': list(), }
    # Retrieve a list of all possible genres
    for movie in range(len(df)):
        # Splitting the genres
        gens = df[f'{column}'].iloc[movie].split('|')
        for gen in gens:
            if gen not in genre_dict[f'{column}']:
                genre_dict[f'{column}'].append(gen)
    # count the number of occurences of each genre
    for genre in genre_dict[f'{column}']:
        count = 0
        for movie in range(len(df)):
            gens = df[f'{column}'].iloc[movie].split('|')
            if genre in gens:
                count += 1
        genre_dict['count'].append(count)

        # Calculate metrics
    data = pd.DataFrame(genre_dict)
    data = data.sort_values(by='count', ascending=False)

    return data


genres = feature_frequency(movies_df, 'genres')


def plot_ratings(count, n=10, color='#4DA017', best=True, method='mean'):
    """
    docstring
    """
    # What are the best and worst movies
    # Creating a new DF with mean and count
    if method == 'mean':
        movie_avg_ratings = pd.DataFrame(train_df.join(
            movies_df, on='movieId', how='left').groupby(['movieId', 'title'])['rating'].mean())
    else:
        movie_avg_ratings = pd.DataFrame(train_df.join(
            movies_df, on='movieId', how='left').groupby(['movieId', 'title'])['rating'].median())
    movie_avg_ratings['count'] = train_df.groupby(
        'movieId')['userId'].count().values
    movie_avg_ratings.reset_index(inplace=True)
    movie_avg_ratings.set_index('movieId', inplace=True)

    # Remove movies that have been rated fewer than n times
    data = movie_avg_ratings[movie_avg_ratings['count'] > count]
    data.sort_values('rating', inplace=True, ascending=False)
    if best == True:
        plot = data.head(n).sort_values('rating', ascending=True)
        title = 'Best Rated'
    else:
        plot = data.tail(n).sort_values('rating', ascending=False)
        title = 'Worst Rated'
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=plot['rating'], y=plot['title'],
                    size=plot['count'], color=color)
    plt.xlabel('Rating')
    plt.ylabel('', fontsize=8)
    plt.tick_params(axis='y', which='both', labelleft=False, labelright=True)
    plt.title(f'Top {n} {title} Movies with Over {count} Ratings', fontsize=14)
    plt.tight_layout()
    plt.show()
