# %%
import pandas as pd

# %% [markdown]
# Pandas is one of the most common and frequently used tools for data analysis. It contains data structures and manipulation tools which make data cleaning, organizing, and analysis easy and efficient. This library is typically used with other libraries like NumPy, SciPy, statsmadels, scikit-learn, and matplotlib.
# 
# While NumPy works best with homogeneous data, pandas is designed to work with tabular, heterogeneous data. The two primary components of pandas are series and DataFrames. You can think of a series as a column of data, and a DataFrame as a collection of series. You can apply operations to a DataFrame as a whole, or to individual columns – this makes it easy to work with various data types.
# 
# You can create DataFrames from scratch, but in the real world we often bring data into a DataFrame from another source – for example, a SQL database, or data files like .csv or .xlsx.
# 
# To install pandas, you can either do so through Anaconda by navigating to your environment and searching through the list of available packages, or directly in Jupyter:
# 
# Pandas is typically imported with the alias pd. This is an industry best practice and I recommend you use this as well.

# %%
df = pd.read_csv('/Users/frieda/Desktop/schulich/MBAN6110S/messy_data.csv')

# %%
df #shown number of rows and number of columns

# %%
# df=df.rename(columns={'Age':'Customer_Age'}) 
# change the title of this column to something more specific: Customer_Age.
# We can do this by using the pandas rename function. The syntax for this is:
#  object.rename(parameters).

# %%
df.info() #Print a concise summary of a DataFrame.

# %%
df.describe(

)# check normal distribution: mean and median is the same 

# %%
df.describe(include = 'all')#unique and top and frequancy is shown 
#astype()can help to chaneg the data type

# %%
df[df['Product'].isnull()]

# %%
df['Product'].isnull()

# %%
df[df['Income']>40000] #64rows so means 65 people income beyond 40000

# %%
summary_D = df[df['Product'] == 'D'].describe()
summary_D 

# %%
type(df[df['Product'] == 'D'].describe())#dataframe means a collection of columns

# %%
print(df['Income'].mean(),df['Income'].median())

# %%
df[df['Income'].isnull()]

# %%
df2 = df.copy()
df2['Income'] = df2['Income'.fillna(df2['Income'].mean())]

# %%
df[df['Product'] == 'D'].head()#head function to only shows the first few lines 

# %%
#df.iloc[50] #filter data by location 
#df.loc[50]
df.set_index('Gender').loc['Female']


