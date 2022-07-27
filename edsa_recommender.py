"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from recommenders.content_based import content_model
from recommenders.collaborative_based import collab_model
from utils.data_loader import load_movie_titles
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from datetime import datetime
from eda import EDA as eda
from utils import data_loader as dl
import plotly
plt.rcParams['figure.dpi'] = 180
warnings.filterwarnings('ignore')
# Igonring warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Style
sns.set(font_scale=1.5)
style.use('seaborn-pastel')

# Custom Libraries

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

path_to_s3 = ('../unsupervised_data/')


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Introduction", "Analysis",
                    "Recommender System", "Solution Overview", 'Contributors']

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png', use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option', title_list[14930:15200])
        movie_2 = st.selectbox('Second Option', title_list[25055:25255])
        movie_3 = st.selectbox('Third Option', title_list[21100:21200])
        fav_movies = [movie_1, movie_2, movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i, j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")

    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection == 'Analysis':

        # Loading EDA Data
        train_df = dl.load_dataframe(
            '../unsupervised_data/unsupervised_movie_data/train.csv', index=None)
        imdb_df = dl.load_dataframe(
            '../unsupervised_data/unsupervised_movie_data/imdb_data.csv', index=None)
        movies_df = dl.load_dataframe(
            '../unsupervised_data/unsupervised_movie_data/movies.csv', index='movieId')

        # Merging movie and train data for EDA
        new_df = train_df.merge(movies_df, on='movieId')
        # Converting time to datetime
        new_df['rating_year'] = new_df['timestamp'].apply(
            lambda timestamp: datetime.fromtimestamp(timestamp).year)
        # Dropping the non converted timestamp
        new_df.drop('timestamp', axis=1, inplace=True)
        # Creating a release year column from title
        new_df['release_year'] = new_df['title'].apply(eda.get_release_dates)
        # Excluding one user with inaccurate data
        eda_df = new_df[new_df['userId'] != 72315]

        st.header("Exploratory Data Analysis")
        st.markdown('Exploratory Data Analysis is an approach to analyzing data sets to summarize their main characteristics, often with visual methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task. \n The visuals shows the kind of data provided, they will be split into the following categories.')
        lst = ['Movies', 'Genre', 'User Interaction']

        for i in lst:
            st.markdown("- " + i)
        st.subheader('Movies')

        # Total movies released per year
        pre_95_releases = pd.DataFrame({'release_year': list(range(1874, 1995)),
                                       'count': eda.get_releases_by_year(new_df, range(1874, 1995))})
        post_94_releases = pd.DataFrame({'release_year': list(range(1995, 2022)),
                                         'count': eda.get_releases_by_year(new_df, range(1995, 2022))})
        overall_movies = pd.DataFrame({'release_year': list(range(1874, 2022)),
                                      'count': eda.get_releases_by_year(new_df, range(1874, 2022))})

        st.subheader('Total  overall movies count   per year')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax1 = overall_movies.groupby('release_year')['count'].sum().plot(
            kind='line', ax=ax, title='Movies released after 1995', color='blue')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('from 1880 to 1995 based of the above graph, we can asssume that from pre 1880 to 1995 the total movies that were released per year are below the 500 in total, which we might not conisder when checking ratings of movies per year. The data is not sufficient to work with, but before that we will check the data of movies pre 1995 and post 1995 to really make sure of our assumption.')

        # Movies released pre 1995 per year
        st.subheader('Total movies count pre 1995 per year')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax1 = pre_95_releases.groupby('release_year')['count'].sum().plot(
            kind='line', ax=ax, color='blue', title='Movies released before 1995')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('The above graphs shows us that our previous assumption was indeed correct, we have under a total of 500 movies per year released pre 1995, We think the reason might have been that less people had access to TVs which made the production companies to really not invest more in movies, techonology was not that advanced before like it is now, only certain type of people where able to have access to TVs, the market was not  that competative.')

        st.subheader('Total movies count post 1995 per year')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax1 = post_94_releases.groupby('release_year')['count'].sum().plot(
            kind='line', ax=ax, title='Movies released after 1995', color='blue')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('The above graph shows us that movie production companies start to increase their production rate for movies, there is a rapid increase of movies per year. That may be because technology waas advancing and more people owned TVs and other sort of alternatives to watch movies.\
            We can see that there was a slide decrease of movie production between 2008 to 2010, we may assume that teh production rate may have been affected by by the 2010 world cup which shifted the focus of many when it came to preparations. \
                The data from post 1995 will really be helpful and useful to visualisations of movie ratings and ratings per year.')

        # Movies Per Genres
        st.subheader('The total number of movies per genre')
        fig = plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            y=eda.genres['genres'], x=eda.genres['count'], palette=("Blues_d"), orient='h')
        plt.title('Number of Movies Per genres', fontsize=14)
        plt.ylabel('genres')
        plt.xlabel('Number of Movies')
        st.pyplot(fig)
        st.markdown('The above graph shows that genre drama appears in most movies, which makes us believe that it might be because Drama delivers the emotional and relational development of realistic characters in a realistic setting. \
            It offers intense character development and tells an honest story of human struggle, which most people can relate to and is why it has so many movies under this genre. \
                It is followed by Comedy, as there is a say that says laughter is best medicine to human kind.')

        st.subheader('Best and Worst Movies by Genre')
        counts = st.number_input(
            'Choose min ratings', min_value=0, max_value=15000, value=10000, step=1000)
        ns = st.number_input('Choose n movies', min_value=5,
                             max_value=20, value=10, step=5)
        st.subheader('Top Best movies by Genre')
        eda.plot_ratings(count=counts, n=ns, color='#4D17A0',
                         best=True, method='mean')
        # plt.tight_layout()
        st.pyplot()
        st.write('By filtering movies with less than 10000 ratings, we find that the most popular movies are unsurprising titles. The Shawshank Redemption and The Godfather unsurprisingly top the list. What is interesting is that Movies made post 2000 do not feature often. Do users have a preference to Older movies?')

        st.subheader('Worst movies by Genre')
        eda.plot_ratings(count=counts, n=ns, color='#4DA017',
                         best=False, method='mean')
        # plt.tight_layout()
        st.pyplot()
        st.write('Obviously, users did not like Battlefield too much and with 1200 ratings, they really wanted it to be known. It is interesting how many sequels appear in the list')

        # Total ratings per year
        st.subheader('The total rating of movies per year')
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax1 = new_df.groupby('rating_year')['rating'].count().plot(
            kind='bar', title='Ratings by year')
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown('The year 2000 and 2016 had the higest total movie rating scores compared to all the other years, like the always 2000s are the problem, we believe that is when the techonoly advanced and people had their chance of expressing how the feel regarding to certain, \
            also again we can assume that it was when they relased the top movies which resulted in higher rating score, the ratings are ranged from 1-5, wwhich makes 5 to be the most top movie and 1 the worst or most disliked movie when rating.\
                We will then look at the ratings to get a better understanding our our ratings movie data.')

    if page_selection == 'Introduction':
        st.header('TEAM CW_6:THE SPACENET TECHNOLOGIES')
        st.image('resources/imgs/company_name.png', use_column_width=True)
        st.subheader('Overview Statement')
        st.markdown('In today\'s technology driven world, recommender systems are critical to ensuring users can make appropriate decisions about the content they engage with daily. Recommender systems help users select similar items when something is being chosen online. Netflix or Amazon would suggest different movies and titles that might interest individual users. In education, these systems may be used to suggest learning material that could improve educational outcomes. These types of algorithms lead to service improvement and customer satisfaction. Current recommendation systems - content-based filtering and collaborative filtering - use difference information sources to make recommendations.')
        st.subheader('About App')
        st.markdown('The app is designed to recommend movies based on content and collaborative filtering, it is capable of accurately predicting movies a user might like based on their preferences.')
        st.info('Collaborative Filtering')
        st.markdown('Collaborative filtering mimics user-to-user recommendations. In other words, If you and your friend have similar tastes, you are likely to make recommendations the other would approve of. This method finds similar users and predicts their preferences as a linear, weighted combination of other user preferences. The limitation is the requirement of a large dataset with active useres who rated a product before in order to make accurate predictions. As a result of this limitation, collaborative systems usually suffer from the "cold start" problem, making predictions for new users challenging. This is usually overcome by using content-based filtering to initiate a user profile.')
        st.info('Content-based Filtering')
        st.markdown('Content-based Filtering makes recommendations based on user preferences for product features. It is able to recommend new items, but is limited by the need for more data of user preference to improve the quality of recommendations.')

    if page_selection == 'Contributors':
        st.image('resources/imgs/meet_team.png', use_column_width=True)
        st.info("Meet the amazing team members that contributed towards this project.")

        st.markdown(
            "<h3 style='text-align: center;'>Mogoboya Caiphus Matibidi</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>App Developer | Data Scientist</p>", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://github.com/caiphus24/" target='_blank'>GitHub</a>""", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://www.linkedin.com/in/caiphus-matibidi-a63786129">LinkedIn</a>""", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: center;'>Mantsi Adelate Kobela</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Data Analyst | Trello Manager</p>", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://github.com/Ade-laide/" target='_blank'>GitHub</a>""", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: center;'>Musa Mashaba</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>App Developer</p>",
                    unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://github.com/codebymusa/" target='_blank'>GitHub</a>""", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: center;'>Ally Monareng</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Data Scientist</p>", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://github.com/ally874/" target='_blank'>GitHub</a>""", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://www.linkedin.com/in/ally-monareng/">LinkedIn</a>""", unsafe_allow_html=True)

        st.markdown(
            "<h3 style='text-align: center;'>Odiaka Chinonso</h3>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center;'>Data Scientist</p>", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://github.com/blaqadonis/" target='_blank'>GitHub</a>""", unsafe_allow_html=True)
        st.markdown("""<a style='display: block; text-align: center;' href="href='https://www.linkedin.com/in/chinonso-odiaka/">LinkedIn</a>""", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
