import pandas as pd


file_path = './hupd_metadata_2022-02-22.feather'
df = pd.read_feather(file_path)

df['filing_date'] = pd.to_datetime(df['filing_date'])
df_2018 = df[df['filing_date'].dt.year == 2018]
print(f"Number of entries in 2018: {len(df_2018)}")

print(df_2018)
print(df.columns.tolist())