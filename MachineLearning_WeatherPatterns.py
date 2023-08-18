# %%
import pandas as pd

# %%
!pwd
%cd /Users/kylenunn/Desktop/Machine-Learning
!pwd

# %%
data = pd.read_csv('/Users/kylenunn/Desktop/Machine-Learning/Weather-2022.csv', encoding= 'unicode_escape')
data

# %%
data.shape
# Dimensions of the data set/matrix is 241 x 22
data.info
# Checking data for number of unique values for each variable
print(data.nunique())
# The variable time of maximum wind gust might not be useful for our purposes so we are going to delete it from the data base.
# Dropping columns or rows using pandas function pop
# Here we have created a python dict in data but we are now going to create is as a pandas dataframe:
block = pd.DataFrame(data)
block.pop('Time of maximum wind gust')
block
# We have succesfully removed the 10th column within the block dataframe
# Now we have to change the names of the column headings to make maniuplation of the data easier:

block.rename(columns = {'Unnamed: 0' : 'Month', 'Minimum temperature (°C)' : 'MinTemp', 'Maximum temperature (°C)': 'MaxTemp',
                       'Rainfall (mm)' : 'Rainfall', 'Evaporation (mm)' : 'Evaporation', 'Sunshine (hours)' : 'Sunlight', 'Speed of maximum wind gust (km/h)' : 'WindGustSpeed', '9am Temperature (°C)' : 'Temp9am',
                       '9am cloud amount (oktas)' : 'Cloud9am', '9am wind direction' : 'WindDir9am', '9am wind speed (km/h)' : 'WindSpeed9am',
                       '9am MSL pressure (hPa)' : 'Pressure9am', '3pm Temperature (°C)' : 'Temp3pm', '3pm relative humidity (%)' : 'Humidity3pm',
                       '3pm cloud amount (oktas)' : 'Cloud3pm','3pm wind direction' : 'WindDir3pm','3pm wind speed (km/h)' : 'WindSpeed3pm',
                       '3pm MSL pressure (hPa)' : 'Pressure3pm', 'Direction of maximum wind gust' : 'WindGustDir'}, inplace = True)
# I can't seem to get 'direction of maximum speed gust' colum heading to change.. look into further..


# %%
print(block.nunique())
# Printed out number of unique quantity within each variable column


# %%
import numpy as np
# Obtain Frequency table in pandas python using value_count() function and crosstab() function
# groupby() count function is used to get the frequency count of the dataframe
# two way frequency table using crosstab() function
# two way frequency of table using proportion / row proportion and column proportions.
block['Month'].value_counts()
block['Date'].value_counts()
block['MinTemp'].value_counts()
block['MaxTemp'].value_counts()
block['Rainfall'].value_counts()
block['Evaporation'].value_counts()

# %%
### Not letting me do anything to or run code with the 'Direction of maximum wind gust' column
### 9AM RELATIVE HUMIDITY (%) not appearing within data frame.. Look into.


# %%
block['Sunlight'].value_counts()
block['WindGustSpeed'].value_counts()
block['Temp9am'].value_counts()
block['Cloud9am'].value_counts()
block['WindDir9am'].value_counts()
block['WindSpeed9am'].value_counts()
block['Pressure9am'].value_counts()
block['Temp3pm'].value_counts()
block['Humidity3pm'].value_counts()
block['Cloud3pm'].value_counts()
block['WindDir3pm'].value_counts()
block['WindSpeed3pm'].value_counts()
block['Pressure3pm'].value_counts()

# %%
block['WindSpeed9am'].value_counts()
# This variable column does indeed have 25 instances of "Calm"
block["WindSpeed9am"].replace({"Calm": 0}, inplace=True)
print(block)

# %%
block['WindSpeed9am'].value_counts()
# Value counts of 'Calm' have been changed to 0.
# I want to see if the new value of 0 is int64,float64, or some other object.

# %%
# An easier way to find the frequency table for all columns is the put them all in one code
block[['WindSpeed9am', 'Month','WindDir3pm', 'WindSpeed3pm', 'Pressure3pm', 'WindDir9am', 'Pressure9am',
      'Temp3pm', 'Humidity3pm', 'Cloud3pm']].value_counts()
# This probably isn't the best way to do this actually becuasea there are so many different numbers and values in this data set...


# %%
block['Date'].value_counts() # Looks good.. only one value per data
block['MinTemp'].value_counts() # All good
block['MaxTemp'].value_counts() # All good
block['Rainfall'].value_counts() # All good

# %%
block['Evaporation'].value_counts() # Values not showing up..
block['Sunlight'].value_counts()
block['WindGustSpeed'].value_counts() # Looks fine
block['Temp9am'].value_counts() # Degrees celsius
block['Cloud9am'].value_counts() # Looks good
block['WindDir9am'].value_counts() # Good
block['Pressure9am'].value_counts() # Fine
block['Temp3pm'].value_counts() # Good
block['Humidity3pm'].value_counts() # Looks fine
block['Cloud3pm'].value_counts() # Cool
block['WindDir3pm'].value_counts() # Fine
block['WindSpeed3pm'].value_counts() # This variable has the same problem as variable 'WindSpeed9am'
block['Pressure3pm'].value_counts() # Good

# Now we will fix the WindSpeed3am column value just as we did the WindSpeed9am


# %%
block["WindSpeed3pm"].replace({"Calm": 0}, inplace=True)
block['WindSpeed3pm'].value_counts()

# %%
# Now we need to check and see if our data has been correctly important. Namely, we need to check and see if intergers, objects, factors, and characters match with their corresponding variable.
block.dtypes

# %%
# block['WindDir9am'] = block['WindDir9am'].astype('float64')
block['WindDir9am'].value_counts() # Good
# The reason you couldn't convert 'WinDir9am' into numeric is becuase they aren't numeric values.
block['WindSpeed9am'] = block['WindSpeed9am'].astype('float64')
block['WindSpeed3pm'] = block['WindSpeed3pm'].astype('float64')

block.dtypes
# Using float64, we have successfully converted the objects in WindSpeed9 and WindSpeed3 to numeric values.

# %%
# We can create new dataframe columns using existing ones
# We can do this by using several methods:
# apply(), numpyselect(), and loc property


# %%
def categorise(row):
    if row['Rainfall'] > 1:
        return '1'
    elif row['Rainfall'] <= 1:
        return '0'
    
block['RainToday'] = block.apply(lambda row:categorise(row), axis=1)

block.head()
# We have succesfully created a new column called 'RainToday' with data values from the already existing column 'Rainfall'

# %%
# So we are now going to move RainToday column up by one and and create it with a RainTomorrow column
block['RainTomorrow'] = block['RainToday'].shift(-1)
block

# %%
# Now I want to make sure there is an equal number of values within each of the columns.. obviously there might be one less value in teh RainTomorrow column
block.count()
# It looks like evaporation and sunlight were made up of NaN values, so we are going to omit those from the block dataframe.
# It looks like Direction of Max Wind Gust, WindGustSpeed, and WindDir3pm, RainTomorrow have 240 values.
# We can remove the entire row for these values since the overall effect change of the dataset would only be 4 missing rows which relatively isn't that much.
# WindDir9am, Cloud9am, and Cloud3pm and substantially less totaling 216, 167, and 184 respectively.
# In conclusion, we are just going to remove the Evaporation and Sunglight variables and keep the rest but realize some columns don't have the same values as the others.


# %%
block.drop(['Evaporation', 'Sunlight'], axis=1)
# Sunglight and Evaporation have been removed.
# For the purposes of this project, I don't want to get rid of the missing values in the columns so I will just set them to the average/mean of the values that were in the column.

# %%
# For out purposes, the data set has been cleaned and we will move on with some EDA.

# %%
# Chi Square test lets us see if there is a difference between the proportions of observations between two populations
# I want to see if the average Maximum Temp is significantly different from the average Maximum temps we would find elsewhere.
# Chi Square Statistic formula : observed-expected**2/expected
# Two sample t-test allows us to see if the means from two different populations are equal.
# We could also do a Chi Square Test of Independed to see if two categorical variables are independent from one another.
# ANOVA allows us to compare several different sample means at the same time.
# We can use a paired t test to compare the same popoulation means at different times.




# %%
# I want to see if there is a signicant difference between WindSpeed9am and WindSpeed9am using a t-test.
# I want to see if RainToday has any influence over whether it will RainTomorrow
# I want to see there is a correlation between Temperature and Humidity. Namely, see if there is a correlation between variables Temp3pm and Humidity3pm

# Binary outcomes are those that can take on only two outcomes, namely 0 and 1.


# %%
# I want to see there is a correlation between Temperature and Humidity. Namely, see if there is a correlation between variables Temp3pm and Humidity3pm

# Covariance:
block.head()
newblock = block[['Temp3pm', 'Humidity3pm']]
newblock
newblock.head()
newblock.corr()

# Interestingly enough, there is a moderately negative correlation between temperature and humidity at 3pm.
# Many times, I feel like humidity is associated with higher temperatures. 
# In the case of this correlation, it is weaker than I expected.



# %%
# I want to see if there is a signicant difference between the means of WindSpeed9am and WindSpeed3am using a t-test.
# This will tell us whther there is a statistically different wind speed between 9am and 3pm.
# Before we perform the test, we need to decide if we’ll assume the two populations have equal variances or not.
# As a rule of thumb, we can assume the populations have equal variances if the ratio of the larger sample variance to the smaller sample variance is less than 4:1. 
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import math

from scipy.stats import ttest_rel
a = block['WindSpeed9am']
b = block['WindSpeed3pm']

ttest_rel(a,b)
# Based on the pvalue and test statistic, it appears that the mean wind speed between 9am and 3pm are statistically significant.




# %%
pip install pingouin


# %%
import pingouin as pt
pt.ttest(a,b,
        paired=True)

# %%
# I want to see if RainToday has any influence over whether it will RainTomorrow
# We have 4 different changes we need to account for:
# Rain-Rain, Rain-NoRain, NoRain-Rain, NoRain-NoRain
# Based on the data, what is the probability it rains tomorrow given that it rains today..
# What is the prob it rains tomorrow given that it doesn't rain today
# What is the prob it doesn't rain tomorrow given that it doesn't rain today
# What is the prob it doesn't rain tomorrow given that it rains today?...

# We need to run a markov chain in python
# Need to get frequency distribution of 0-0,0-1,1-1,1-0.
block.groupby(["RainToday", "RainTomorrow"]).size()


# %%
A = np.array([[.6541,.1417],[.1417,.0625]])

# %%
157+34+34+15
157/240
34/240
15/240


# %%
.6541,.1417,.1417,.0625
# P(RainToday):
(34+15)/240

# %%
A
# Now we are going to start with a random walk.

# %%
states = ["NoRain", "Rain"]
transitionName = [["NN","NR"],["RN","RR"]]
transitionMatrix = [[.82,.18],[.69,.31]]
import random as rm

def weather_forecast(days):
    # Choose the starting state
    weatherToday = "NoRain"
    print("Start state: " + weatherToday)
    # Shall store the sequence of states taken. So, this only has the starting state for now.
    weatherList = [weatherToday]
    i = 0
    # To calculate the probability of the weatherList
    prob = 1
    while i != days:
        if weatherToday == "NoRain":
            change = np.random.choice(transitionName[0],replace=True,p=transitionMatrix[0])
            if change == "NN":
                prob = prob * 0.82
                weatherList.append("NoRain")
                pass
            elif change == "NR":
                prob = prob * 0.18
                weatherToday = "Rain"
                weatherList.append("Rain")
        elif weatherToday == "Rain":
            change = np.random.choice(transitionName[1],replace=True,p=transitionMatrix[1])
            if change == "RR":
                prob = prob * 0.69
                weatherList.append("Rain")
                pass
            elif change == "RN":
                prob = prob * 0.31
                weatherToday = "NoRain"
                weatherList.append("NoRain")
            i += 1
        print("Possible states: " + str(weatherList))
        print("End state after "+ str(days) + " days: " + weatherToday)
        print("Probability of the possible sequence of states: " + str(prob))
# Function that forecasts the possible state for the next 2 days
weather_forecast(2)

# We can see a plethora of probabilities that relate to what we could expect.
# I still have yet to figure out a simple markov chain for the above data but we will save that for a later date.

# %%
%matplotlib inline
import matplotlib.pyplot as plt
block.hist(bins=50, figsize=(20,15))
plt.show()

# Here are the distributions for all numerically valued columns.

# %%
# Using value_counts to create histograms of categorical variables.
block['WindDir3pm'].value_counts().plot(kind='bar')
block['Month'].value_counts().plot(kind='bar')
block['Cloud9am'].value_counts().plot(kind='bar')

# %%
## End of Project 1 ##