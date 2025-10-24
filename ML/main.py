import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ======================================
# Reading csv and manipulating with numpy
# =======================================

df = pd.read_csv('./csv/data.csv')
print(df) # See if the data was loaded

df['Revenue'] = df['price'] * df['sales_last_month'] # Creating the revenue column by multiplying the two columns of price and sales_last_month
df['Above_avg_price']= (df['price'] > np.mean(df['price'])).astype(int)
df['Above_avg_sales_lt_month']= (df['sales_last_month'] > np.mean(df['sales_last_month'])).astype(int)


print(df) # Seeing after eventual changes

per_category = df.groupby('category').agg(
    Receita_por_categoria = ('Revenue','sum'),
    Receita_média_categoria = ('Revenue','mean'),
    Vendas_por_categoria = ('sales_last_month','sum'),
    Média_venda_categoria = ('sales_last_month','mean'),
    DiscontoMedio_por_categoria = ( 'discount' , 'mean'),
    Avg_rating_categoria = ('rating', 'mean')
)

per_subcategory = df.groupby('subcategory').agg(
    Receita_por_subcategoria = ('Revenue','sum'),
    Media_receita_subcategoria = ('Revenue','mean'),
    Vendas_por_subcategoria = ('sales_last_month','sum'),
    Média_venda_subcategoria = ('sales_last_month','mean'),
    DiscontoMédio_por_subcategoria = ('discount','mean'),
    Average_rating_subcategoria = ('rating', 'mean')
)

# =========================================================
# Creating a linear regression to predict the relation between price and sale from the given df
# =========================================================

X = df[['price']]
y = df['sales_last_month']

X_train, X_test , y_train , y_test = train_test_split( # Splitiong training
    X,y, test_size = 0.1, random_state = 42
)

model = LinearRegression().fit(X_train,y_train)
y_pred= model.predict(X_test)

# =====================
# Evaluting the Model
# =====================

print('MAE:',mean_absolute_error(y_test,y_pred))
print('MSE:', mean_squared_error(y_test,y_pred))


# =============
# Exporting files
# ================

per_category.to_csv('./csv/dataPerCategory.csv', index = True)
per_subcategory.to_csv('./csv/dataPerSubCategory.csv' , index = True)

# ===================
# Visualizing the data
# ===================



# Sort X_test for a proper line plot
sorted_idx = np.argsort(X_test['price'].values)
X_test_sorted = X_test['price'].values[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize=(10,7))
plt.scatter(X_test, y_test, label='Actual Data', color='blue')
plt.plot(X_test_sorted, y_pred_sorted, label='Predicted Data', color='red', linewidth=2)
plt.title('Machine Learning Curve')
plt.xlabel('Price')
plt.ylabel('Sales Last Month')
plt.legend()
plt.show()