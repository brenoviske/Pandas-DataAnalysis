import numpy as np # For vectorized operations
import matplotlib.pyplot as plt # For data visualization
import pandas as pd # Fot Data Analisys

def Monthly_sales():
    # Making a data analysis for an ficctiocional e-commerce
    df = pd.read_csv('monthly_sales.csv')

    # Changing the DataFrame and adding new columns

    df['Average_Month'] = np.mean(df[['Product A' , 'Product B' , 'Product C']], axis = 1)

    df['Sum_Month'] = np.sum(df[['Product A' , 'Product B' , 'Product C']] , axis = 1 )
    
    # Making a graphic for Data Visualization

    plt.plot(df['Month'], df['Product A'] ,label = 'Product A')
    plt.plot(df['Month'], df['Product B'] , label = 'Product B')
    plt.plot(df['Month'] , df['Product C'] , label = 'Product C')
    plt.xlabel('Months')
    plt.ylabel('Sales')
    plt.xticks(rotation = 45)
    plt.title('Months x Sales Graphic')
    plt.legend()


    # Displaying the graphics and the DataFrame

    print(df)
    plt.show()

    #Exporting the Graphic to a new csv file

    df.to_csv('monthly_sales_new.csv',index=False)


if __name__ == '__main__': Monthly_sales()