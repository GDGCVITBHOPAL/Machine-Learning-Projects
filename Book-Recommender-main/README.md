# Book-Recommender
A Machine Learning Project that makes the use of KNN to recommend books to the users based off of average ratings of the books in the Data, the language they are written in, and the total rating counts for that book.

# Files In the Folder
1) Book Recommender -  Contains the actual Recommender System. <br>
2) books.csv - The data that was used to make the Model.

# Tools used 
<img src="https://img.shields.io/badge/python%20-%2314354C.svg?&style=for-the-badge&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/pandas%20-%23150458.svg?&style=for-the-badge&logo=pandas&logoColor=white" /> <img src="https://img.shields.io/badge/Jupyter%20-%23F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />

# Objective of the Project 
The objective of this Recommender system was to Recommend Books to the user based on the average ratings that the books have recieved, the number of rating counts and the languages these books are written in.

# Steps involved in making the Recommender System
1) We start off by importing and taking a look at our data. We have in total 10 columns to work with. These columns include - the authors, book title, publishers and many other details that might be relevant to the books. More information on these columns is provided in the main notebook. <br>
2) We then check some basic information such as checkig for any null data present, checking for maximum and minimum values of each column and finding the data types of each column. This step helps us in knowing a little bit more about our data and makes our further processes easier. <br>
3) Next, we do some visualizations on on the columns in our data. We find the top 10 athors, some highly rated authors, the top 10 publishers and try and find relations between the average ratings of the books and number of ratings that these books have recieved. These helps us in having a better understanding of our data and also helps us in selectig the right features to make our model on.<br>
4) We chose average ratings, the rating counts and languages as our features to feedd to our model. It would have been better if we had a category column as well, we could use this to suggest books according to the genres but our data set didn't have these values but we can obtain them externally through scraping. <br>
5) We feed these features to our model which is a KNN based model. So what it does is, a user inputs a name of the book, this book name becomes a data point, then our model looks for other data points that are in the vicinity of this data point to make our Recommendation. <br>
6) We finally define a method to make our predictions and that concludes our whole process of making this recommender system.<br>

# Results
The System actually makes some decent recommendations based on the inputs. These recommendations can be made even better with a category column as well. We have a column called ISBN-13 which we cann use to scrape the Categories of each book in our data but even with our features, the system performs well.<br>

Results when we input one of the Harry Potter books - <br>
![alt text](https://github.com/AM1CODES/Machine-Learning-Projects/blob/main/Book-Recommender-main/Result-1.PNG?raw=true)

Results when we input one of the Lord of the Rings books - <br>
![alt text](https://github.com/AM1CODES/Machine-Learning-Projects/blob/main/Book-Recommender-main/Result-2.PNG?raw=true)
