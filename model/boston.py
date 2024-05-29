import ipp # type: ignore

compModel = {}

df = ipp.pd.read_csv('/Users/nikhilprao/Documents/Data/Boston.csv', index_col=0)

df.describe()

