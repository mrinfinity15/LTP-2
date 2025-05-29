df = pd.read_csv("d:/LTP 2/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1", nrows=5000)

# Show all column names to find the correct one
print(df.columns.tolist())
exit()  # Stop script after printing
import pandas as pd

# Load only first 5,000 rows for speed
df = pd.read_csv("d:/LTP 2/DataCoSupplyChainDataset.csv", encoding="ISO-8859-1", nrows=5000)

# Show all columns to identify correct name
print("🧾 Column Names in the Dataset:\n")
for col in df.columns:
    print(f"- '{col}'")

# Exit after printing
exit()
