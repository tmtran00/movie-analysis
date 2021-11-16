# movie-analysis
While learning Python, I used various Python libraries to extrapolate and analyze data from movies box office data. These libraies include: pandas, matplotlib, seaborn and sklearn.
# Results
![image](https://user-images.githubusercontent.com/93699932/141875462-8b5615e0-00b7-413a-ab6e-fbf3d5e924b5.png)

This graph shows that as budget increase, so do the profit of the film. There are also outliers in most of these graphs due to unpredictability of a movie’s success. For example, in this case, the highest earning action movie is the 2009 James Cameron film Avatar, one of the highest grossing film of all time, which make sense as to why it is so ridiculously high on this graph.  

What’s interesting is that some other genres also share similar inverse relationship between its budget and profitability. For instance, comedy movies. 

![image](https://user-images.githubusercontent.com/93699932/141875671-69132f6f-e48b-4985-8bea-c7523abf2649.png)

Although these movies spend more money on production, the profit margin however does not seem to increase but rather decreasing. 
# Machine Learning Results

I wanted to use these data, feed it to the sklearn library and see if I can predict a movie's success given certain criteria from the movies. From the data, I separated all the information about the production of movie given by the data. This included the studios making the movie, the one-word summary of the movie, the genre, budget, critics and audience Rotten Tomatoes scores, number of theatres playing the movies, and the year of production. Those are then my features and my label will be the world gross. I then split the data 80:20, 80% to train and 20% to test my result. Using the mean square error (MSE) function from sklearn, I test the effectiveness of the model. The result given by the MSE function ranged in the ten-thousands. Meaning that the model is not an accurate predictor the success of these movies.  

# Future improvements

Although the my process of predicting of movies success was not accurate, it did help me further understand the limitation of the sklearn library and what else I can do to improve the accuracy of the predictions.
I could find a more extensive set of data with more columns to build out my layers and nodes. That way we can take into account more factors than can affect the outcome of a movie's success.
I also have to take into consideration the ethics of building a predictor like this as it may affect real people whose livelihood is based on the product that my program is trying to predict.




