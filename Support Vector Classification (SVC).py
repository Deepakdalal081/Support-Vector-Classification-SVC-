#1. Define a Classification Task
# Importing necessary library
import pandas as pd
import numpy as np
import datetime as dt 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')


#2. Read the Dataset
end_date = dt.date.today()
start_date = end_date - dt.timedelta(days = 100)

data = yf.download("AXISBANK.NS", start=start_date, end=end_date, interval="1D")

data["Return"] = np.log(data["Close"]/data["Close"].shift(1))

#3. Generate Target Values
data["Target"] = np.where(data["Open"].shift(-1)> data["Open"], 1, -1)

data["Volume"] = data["Volume"].shift(1)

#4. Feature Selection
plt.figure(figsize =(10,6))
sns.scatterplot(x= data["Return"], y = data["Volume"])


#5. Feature Extraction
feature_list = []

#- Rolling standard deviation
for i in range (5,25,5):
    colum_name = "SD_" + str(i)
    data[colum_name] = data["Close"].rolling(window= i).std()
    feature_list.append(colum_name)
 
#- Rolling moving average of close price
for i in range(5,25,5):
    colum_name = "MA_" + str(i)
    data[colum_name] = data["Close"].rolling(window = i).mean()
    feature_list.append(colum_name)
    
#- Rolling percentage change
for i in range (5,25,5):
    colum_name = "pct_" + str(i)
    data[colum_name] = data["Close"].pct_change().rolling(window = i).sum()
    feature_list.append(colum_name)

#- Rolling moving average of volume
colum_name = "Vol_4"
data[colum_name] = data["Volume"].rolling(4).mean()
feature_list.append(colum_name)

#- Difference between close and open
colum_name = "CO"
data[colum_name] = data["Close"]- data["Open"]
feature_list.append(colum_name)

#Drop all the NaN values
data.dropna(inplace = True)

data[feature_list + ["Target"]]

#6. Generate Train-Test Datasets
from sklearn.model_selection import train_test_split

X = data[feature_list].iloc[:-1]
Y = data.iloc[:-1]["Target"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=.75, shuffle=False)


#7. Feature Scaling
#Ml requie data in the normalized form 

from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

X_train_scalled = scalar.fit_transform(X_train)
X_test_scalled = scalar.transform(X_test)

X_train_scalled = pd.DataFrame(X_train_scalled, columns= X_train.columns)
sns.pairplot(X_train_scalled[["SD_15", "MA_10", "pct_5","Vol_4","CO"]])

#8. Build Model
from sklearn.svm import SVC
model = SVC(kernel="poly", random_state=1)


#9. Train Model
model.fit(X_train_scalled, y_train)

#10. Predict
y_pred_train = model.predict(X_train_scalled)

print("The train data Accuracy is = ", model.score(X_train_scalled, y_train))

x_pred_test = model.predict(X_test_scalled)

print("The test data Accuracy is = ", model.score(X_test_scalled, y_test))
