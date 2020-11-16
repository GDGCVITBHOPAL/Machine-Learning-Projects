# Twitch Top Streamers Analysis

# Objectives  
The objective is to predict the followers gained by a streamer on Twitch during the past year of streaming. <br>

# Tools Used
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" /> <img src="https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" /> <img src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />

# Data Set used
The data set used is a custom made data set compiled by  taking the stats from different websites.

# Contents of our Data set
1) Channel - The name of the Streamer's channel <br>
2) Watch time(minutes) - The amount of time people have watched a particular streamer. <br>
3) Stream time(minutes) - The total amount of time the streamer streamed in the past year. <br>
4) Peak viewers - The maximum number of viewers the streamer had during his stream in the past year. <br>
5) Average viewers - The average number of viewers a streamer had while streaming. <br>
6) Followers - Number of Followers a Streamer had on Twitch. <br>
7) Followers gained - The amount of followers a particular streamer gained in a year on Twitch. <br>
8) Views gained - The amount of views a streamer gained in a year. <br>
9) Partnered - Whether the streamer is Twitch Partnered or not. <br>
10) Mature  -Whether the streams are 18+ or not. <br>
11) Language - The language used by a streamer during the stream. <br>


# Steps involved in making the model 
1) Importing our data. <br>
2) Checking for null values and finding mean of different columns, their min and max values, and getting information about different columns of our data.<br>
3) Visualizing our data in order to find out the best columns to use as features for our model. <br>
4) Make a copy of original data to make the changes and seperate out the columns that we will use as our features. <br>
5) Using these features, we will train our Linear Regression Model.<br>
6) Concluding our analysis by testing the model with some random user input.<br>

# Result  
Our model was able to predict a certain number of Followers gained but the error can be decresed further by optimizing it. We can also try and use a few features like whether the streamer is twitch partnered or not,etc. <br>
Our Result: <br>

![alt text](https://github.com/AM1CODES/TwitchTopStreamers-Analysis/blob/main/Twitch-Result.PNG?raw=true)
