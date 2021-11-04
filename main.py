import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
sns.set()


def optimal_budget(data):
    """
    Make a budget v. worldgross and budget v. profitability regression plot
    or each unique genre in data
    para:
    data = DataFrame
    """

    # find get unique genres
    data = data.dropna()
    genres = data['Genre'].unique()

    # loop through all genres, plot budget v. world gross for each genre
    for i in range(len(genres)):
        fig, ax = plt.subplots()
        sns.regplot(x='Budget',
                    y='WorldGross', data=data[(data['Genre'] == genres[i])])
        plt.title(genres[i] + ' Movies Budget and World Earnings')
        plt.xlabel('Budget (in million)')
        plt.ylabel('World Grossing(in million)')
        fig.savefig(fname=str(genres[i])+'.png', bbox_inches='tight')

    # loop through all genres, plot budget v. profitability gross for each
    for i in range(len(genres)):
        fig, ax = plt.subplots()
        sns.regplot(x='Budget',
                    y='Profitability', data=data[(data['Genre'] == genres[i])])
        plt.title(genres[i] + ' Movies Budget and Profitability')
        plt.xlabel('Budget (in million)')
        plt.ylabel('Profitability')
        fig.savefig(fname=str(genres[i])+'_prof.png', bbox_inches='tight')


def best_action(data):
    """
    Find highest world grossing Action movie

    para:
    data = Dataframe
    """
    filtered = data[data['Genre'] == 'Action']
    max = filtered['WorldGross'].idxmax()
    movie = filtered.loc[max]
    return movie['Movie']


def gross_scatter(data):
    """
    Create budget v. world gross scatter plot for Horror and Crime

    para:
    data = DataFrame
    """
    data = data.dropna()
    # horror
    fig1, ax = plt.subplots()
    sns.relplot(x='Budget',
                y='WorldGross',
                data=data[data['Genre'] == 'Horror'], kind='scatter')
    plt.title('Horror Movies Budget and World Earnings')
    plt.xlabel('Budget (in million)')
    plt.ylabel('World Grossing(in million)')
    plt.savefig(fname='Horror_SCATTER.png', bbox_inches='tight')

    # Crime
    fig2, ax = plt.subplots()
    sns.relplot(x='Budget',
                y='WorldGross',
                data=data[data['Genre'] == 'Crime'], kind='scatter')
    plt.title('Crime Movies Budget and World Earnings')
    plt.xlabel('Budget (in million)')
    plt.ylabel('World Grossing(in million)')
    plt.savefig(fname='Crime_SCATTER.png', bbox_inches='tight')


def ratings(data):
    """
    Create critics Rotten Tomatoes v. opening weekend box office
            audience Rotten Tomatoes v. opening weekend box office

    para:
    data = DataFrame
    """
    # Critics v. Opening weekend
    data = data.dropna()
    fig1, ax1 = plt.subplots()
    sns.regplot(data=data, x='RottenTomatoes', y='OpeningWeekend')
    plt.title('Critics RottenTomatoes Rating on Opening Weekend Earnings')
    plt.xlabel('Rotten Tomatoes Rating')
    plt.ylabel('Opening Box Office (in millions)')
    fig1.savefig(fname='rating.png', bbox_inches='tight')

    # Audience v. Opening weekend
    fig2, ax2 = plt.subplots()
    sns.regplot(data=data, x='AudienceScore', y='OpeningWeekend')
    plt.title('Audience RottenTomatoes Rating on Opening Weekend Earnings')
    plt.xlabel('Audience Rating')
    plt.ylabel('Opening Box Office (in millions)')
    fig2.savefig(fname='rating2.png', bbox_inches='tight')

    # Comedy: critics v. opening weekwend
    filtered = data[data['Genre'] == 'Comedy']
    fig3, ax3 = plt.subplots()
    sns.regplot(data=filtered, x='RottenTomatoes', y='OpeningWeekend')
    plt.title('Critics RottenTomatoes Rating on Opening Weekend for Comedy')
    plt.xlabel('Rotten Tomatoes Rating')
    plt.ylabel('Opening Box Office (in millions)')
    fig3.savefig(fname='comedy_rating.png', bbox_inches='tight')

    # Comedy: audience v. opening weekend
    filtered = data[data['Genre'] == 'Comedy']
    fig3, ax3 = plt.subplots()
    sns.regplot(data=filtered, x='AudienceScore', y='OpeningWeekend')
    plt.title('Audience RottenTomatoes Rating on Opening Weekend for Comedy')
    plt.xlabel('Rotten Tomatoes Rating')
    plt.ylabel('Opening Box Office (in millions)')
    fig3.savefig(fname='comedy_rating2.png', bbox_inches='tight')


def predict_success(data):
    """
    Machine learning model to predict the sucess of movies

    using movie production info as features
    world grossing as labels

    para:
    data = DataFrame
    """
    data = data.dropna()
    filtered = data[['LeadStudio', 'RottenTomatoes',
                    'Genre', 'AudienceScore', 'Story',
                     'TheatersOpenWeek', 'Budget', 'Year']]
    X = pd.get_dummies(filtered)
    y = data['Profitability']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)
    predict = model.predict(x_test)

    return(mean_squared_error(y_test, predict))


def markets(data):
    """
    Create genres v. domestic and foreign gross bar charts

    Use top 5

    para:
    data = DataFrame

    """
    data = data.dropna()
    domes = data.groupby('Genre')['DomesticGross'].sum()
    fore = data.groupby('Genre')['ForeignGross'].sum()
    domes = domes.sort_values(ascending=False)
    domestic_graph = sns.catplot(
                                x='DomesticGross',
                                y='Genre',
                                data=domes.reset_index()[0:5], kind='bar')
    plt.title('Top 5 Genres Domestic Gross')
    plt.xlabel('Box Office Earnings (in millions)')
    plt.ylabel('Genres')
    domestic_graph.savefig(fname='Domestic.png', bbox_inches='tight')

    fore = fore.sort_values(ascending=False)
    fore_graph = sns.catplot(x='ForeignGross',
                             y='Genre',
                             data=fore.reset_index()[0:5], kind='bar')
    plt.title('Top 5 Genres Foreign Gross')
    plt.xlabel('Box Office Earnings (in millions)')
    plt.ylabel('Genres')
    fore_graph.savefig(fname='Foreign.png', bbox_inches='tight')


def how_many(data, genre):
    """
    Return number of movie in each given genre

    para:
    data = DataFrame
    genre = genre of movie
    """

    data = data.dropna()
    filtered = data[data['Genre'] == genre]
    return len(filtered)


def total_movies(data):
    """
    Print number of movies for each genres
    """
    data = data.dropna()
    for i in data['Genre'].unique():
        print(i + ' ' + str(how_many(data, i)) + ' movies')


def main():
    data = pd.read_csv('Project\\HollywoodMovies.csv')
    total_movies(data)
    print(best_action(data))
    gross_scatter(data)
    optimal_budget(data)
    ratings(data)
    print(predict_success(data))
    markets(data)


if __name__ == '__main__':
    main()
