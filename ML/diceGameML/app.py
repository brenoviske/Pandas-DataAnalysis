import numpy as np
import pandas as pd

samples = abs(int(input('Type the number of rounds you want to be analyzing:')))
num = abs(int(input('Type the number that you want to be your paramter for analysis:')))
dice = np.random.randint(1,7, size = (samples,2))

# ========================
# Generating the DataFrame
# ========================

df = pd.DataFrame(dice, index = [f'Round{i}' for i in range(1,samples+1)], columns = ['Player1','Player2'])
df['RoundSum'] = np.sum(df,axis=1) 
df['RoundAverage'] = df[['Player1','Player2']].mean( axis = 1)
df['RoundDeviation'] = round(df[['Player1','Player2']].std( axis = 1),2)
df['Above_num'] = (df['RoundSum'] > num).astype(int)

df.to_csv('dicegame_analysis.csv', index = True) # Exporting to the .csv folder

# OUTPUTING THE INFORMATION 
print(df)
print(f'Round with sum above {num}:{sum(df['Above_num'])}')
