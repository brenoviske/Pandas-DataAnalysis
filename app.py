# Creating a code to make a fast data analysis by using Python, just as an example
import pandas as pd # importing the Pandas library

# reading the csv file ( can be a html file, excel file) and creatinfg the DataFrame

df = pd.read_csv('vendas.csv')

#Creating new columns

df['Revenue'] = df['Price'] * df['Amount']
average = df['Revenue'].sum() # generating the average revenue

df['Above_average'] = df['Revenue'] > average

#Filtering the Data

# Filtering by Employee

filtered_by_employee = df.groupby('Employee').agg(
    Average_revenue_Employee = ('Revenue','mean'),
    Revenue_per_employee = ('Revenue','sum')

)

# Filtering by Product

filtered_by_procut = df.groupby('Product').agg(
    Revenue_per_prodcut = ('Revenue','sum'),
    Average_per_product = ('Revenue','mean'),
    Amount_per_prodcut = ('Revenue','count')
)


# Exporting the Data ton new csv files

filtered_by_employee.to_csv('filtered_data_employee.csv' , index = True)
filtered_by_procut.to_csv('filtered_data_product.csv',index = True)
df.to_csv('vendas_att.csv',index=False)