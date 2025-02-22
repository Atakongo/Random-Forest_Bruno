import pandas as pd

#file_path
file_path = 'data_set.xlsx'

# Read the .xlsx file into a DataFrame
df = pd.read_excel(file_path)

# Display the DataFrame
print(df.head())


# Define the column names
split_column = 'MSP-IDs NP Produkt'  # the last column of the first split

# Find the index of the split column
split_index = df.columns.get_loc(split_column) + 1  # +1 to get the next column

# Split the dataframe
df1 = df.iloc[:, :split_index]  # Columns from 'Titel NP Produkt' to 'MSP-IDs NP Produkt'
df2 = df.iloc[:, split_index:]  # Remaining columns

def getDF1():
    return df1

def getDF2():
    return df2