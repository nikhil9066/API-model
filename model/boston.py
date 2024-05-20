import ipp # type: ignore

compModel = {}

df = ipp.pd.read_csv('/Users/nikhilprao/Documents/Data/Boston.csv', index_col=0)
df.columns

df.isnull().sum()

df.reset_index(drop=True)
type(df)
df.describe()
df.head()
df.info()
data = df
type(data)

corr_mat=data.corr()

corr_mat
type(corr_mat)

