from sklearn.datasets import load_iris 
import pandas as pd

iris = load_iris()

df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

df['species'] = iris.target 

species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_names)

df.to_csv('data/iris.csv', index=False)

print("✓ Dataset saved to data/iris.csv")
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nSpecies distribution:")
print(df['species_name'].value_counts())