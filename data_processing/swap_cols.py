import pandas as pd

# Read the CSV file
df = pd.read_csv('power_1778_data.csv')

# Swap the columns (e.g., swapping 'A' and 'B')
columns = list(df.columns)
col1, col2 = 'Power_Level', 'Angle'  # Replace with the actual column names

# Find their positions
idx1, idx2 = columns.index(col1), columns.index(col2)

# Swap the column positions in the list
columns[idx1], columns[idx2] = columns[idx2], columns[idx1]

# Reorder the DataFrame based on the updated column order
df = df[columns]

# Save the result to a new CSV file
df.to_csv('output.csv', index=False)
