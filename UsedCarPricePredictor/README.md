## CarPricePredictor
A used-car price predictor created using Machine-Learning and Web Development.

#### Problem Statement: 
The auto industry is changing rapidly and car prices are only going up. So to speak, new cars are getting costlier each year, making them a very high value purchase for the common man. And quite ironically, the average life span of a car is going down despite the steady rise in prices, which brings in good news for potential used car buyers! Thanks to manufacturers launching newer versions of their models sooner now as compared to a few years ago, more and more modern cars are now entering the used car market, which makes it easy for you to make a good buy without having to bust your wallet. <br>
<b>And as the market for used-cars is increasing, the number of sellers is also increasing. Keeping this in mind, I have created a web application for sellers, to know what will be the price of their car, in the market.</b>

#### Dataset used:
For solving this problem, I have used the CarDekho dataset, which was avaliable on Kaggle. This is the link: https://www.kaggle.com/shindenikhil/car-dekho-data

#### References Used: 
I have used <a href="https://www.youtube.com/watch?v=p_tpQSY1aTs">this video</a> as a reference. A quick shoutout to <a href="https://www.youtube.com/channel/UCNU_lfiiWBdtULKOw6X0Dig">Krish Naik</a>, whom I have been following since May.<p>He uploads very informative and concise videos for Machine Learning and many other topics in the field of AI.</p>

#### Algorithms used:
After experimenting with a lot of Regression-based algorithms, <b>Random Forest</b> has worked out the best for me in terms of accuracy. If you want an in-depth explaination on random Forest and how do they work, feel free to checkout <a href = "https://towardsdatascience.com/understanding-random-forest-58381e0602d2">this article</a>. 

#### Features used:
Since this is a very basic used-car price predictor, I have only used a handful of features. Some of the features are:
1. Year: The total life-span of the car(currentYear - yearOfPurchasing)
2. Showroom Price: The price of the car when the seller first bought it.
3. Kilometers Driven: The total number of kilometers the car has driven.
4. Previous Owners: The number of owners the car has had.
5. Fuel Type: Specifies if the car runs on Petrol, Diesel or CNG.
6. Dealer/Individual: Specifies if the seller is a dealer or an individual.
7. Transmission Type: Specifies whether the car is a manual one or automatic.

#### NOTE:
"Success is the project that's always under construction": Keeping this quote in mind, I'd like to add many features into this project in the future. So stay updated!

#### To run this on your machine:
1. Make sure you have Python3, Flask installed.
2. In you terminal, cd to the folder you got this project in, and type <b>python run app.py</b>
3. And voila!!

#### P.S: I will be deploying this project fully in the upcoming days, so that you guys don't have to download the project folder everytime.

#### Here are some screenshots of the application: 
![carPred1](https://user-images.githubusercontent.com/55303125/111125944-2c519780-8598-11eb-925b-12ba4cfef0a1.png)
<br>
![carPred2](https://user-images.githubusercontent.com/55303125/111126311-8f432e80-8598-11eb-9b7e-d17dd4f9f99a.png)

