# ---------------- This Script here was made to evalute metrics from a given real DataSet( House pricing in the state of California ) -------------- #

# ------ Importing necessary tools ------- #
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

housing_dataset = fetch_california_housing(as_frame = True)
df = housing_dataset.frame # ( Loading the real dataset into a DataFrame )
print(df) # ( Checking )

x = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']] # ( Gathering the data for x-axis analysis)
y = df['MedInc'] * 100 +  2000 * df['HouseAge']  + df['AveRooms'] - df['AveBedrms'] # ( Making a combinar relation in a way to estime the price of these houses, thos is the synthentic pricing relation )

scaler_x = StandardScaler()
scaler_y = StandardScaler()

# ( Normalizing the Data for mean = 0 and std  = 1)
x = scaler_x.fit_transform(x)
y = scaler_y.fit_transform(y.values.reshape(-1,1))

X_train , X_test , y_train , y_test = train_test_split(
    x,y , random_state = 42 , test_size = 0.1 # ( Notice that this line right here can indeed be changed to adjust the amount of training set that u indeed want to train your model upon)
)

model = LinearRegression().fit(X_train,y_train)
pred = model.predict(X_test)

x_test_original = scaler_x.inverse_transform(X_test)
y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(pred)

# ( Now evaluating the models performance and plotting the data into a x -> y graphic )
# ( For us to indeed see what is the Linear relation and prediction between those x features and the house pricing )

print('Model Accuracy\n')
print('MAE:', mean_absolute_error(y_test,pred))
print('MSE:', mean_squared_error(y_test,pred))
print('R2:',r2_score(y_test,pred))


# ( Ploting the graphic MedInc vs Pricing )
plt.scatter(x_test_original[:,0], y_test_original , label = 'Real Data' , color = 'pink')
plt.scatter(x_test_original[:,0], y_pred_original, label = 'Predicted Data' , color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Med Inc ')
plt.ylabel('Synthetic Pricing ')

plt.legend()
plt.show()


# ( Plotting the graphic HouseAge x Pricing )

plt.scatter(x_test_original[:,1], y_test_original , label = 'Real Data' , color = 'green')
plt.scatter(x_test_original[:,1], y_pred_original, label = 'Predicted Data' , color = 'red')
plt.title('Linear Regression')
plt.xlabel('House Age')
plt.ylabel('Synthetic Pricing ')

plt.legend()
plt.show()

# ( Ploting the graphic AveRooms x Pricing )

plt.scatter(x_test_original[:,2], y_test_original , label = 'Real Data' , color = 'green')
plt.scatter(x_test_original[:,2], y_pred_original, label = 'Predicted Data' , color = 'black')
plt.title('Linear Regression')
plt.xlabel('Average Rooms ')
plt.ylabel('Synthetic Pricing ')

plt.legend()
plt.show()

# ( Ploting the graphic AveBedrms x Pricing )

plt.scatter(x_test_original[:,3], y_test_original , label = 'Real Data' , color = 'purple')
plt.scatter(x_test_original[:,3], y_pred_original, label = 'Predicted Data' , color = 'orange')
plt.title('Linear Regression')
plt.xlabel('Average Bedrooms')
plt.ylabel('Synthetic Pricing ')

plt.legend()
plt.show()

