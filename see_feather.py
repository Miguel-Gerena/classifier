import pandas as pd


file_path = './hupd_metadata_2022-02-22.feather'
df = pd.read_feather(file_path)

# df['filing_date'] = pd.to_datetime(df['filing_date'])
# df_2018 = df[df['filing_date'].dt.year == 2018]
# print(f"Number of entries in 2018: {len(df_2018)}")

# print(df_2018)
print(df.columns.tolist())

# start_date = pd.to_datetime('2015-01-01')
# end_date = pd.to_datetime('2017-12-31')
# filtered_df = df[(df['filing_date'] >= start_date) & (df['filing_date'] <= end_date)]
# print(filtered_df)