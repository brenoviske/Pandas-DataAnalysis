# FOR MACHINE LEANRING ALGORITHMS

from app import df
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# FOR DATA VISUALIZATION 

import matplotlib.pyplot as plt

# Trying to relate the player values to the average round score pontuation through a Linear Regression

X = df[['Player1']] # or it could .values.reshape(-1,1) using numpy
y = df['RoundAverage']

X_train , X_test ,  y_train , y_test = train_test_split(
    X,y,test_size = 0.2, random_state = 42
)

model = LinearRegression().fit(X_train,y_train)
y_pred = model.predict(X_test)

# Evaluating the model , it will tend to have high deviations since the gathered data is entirely random 
# so the ML model is essencially guessing results 

print('Evaluating model efficiency:')
print('MSE:', mean_squared_error(y_test,y_pred))
print('MAE:', mean_absolute_error(y_test,y_pred))


plt.figure( figsize = (10,6) )
plt.scatter(X_test, y_test, label = 'Real Data', color = 'blue')
plt.plot(X_test,y_pred,label = 'Linear Regression', color = 'red')
plt.title('Linar Regression with Sklearn')
plt.xlabel('Player scoring')
plt.ylabel('Average per Round')

plt.legend()
plt.show()