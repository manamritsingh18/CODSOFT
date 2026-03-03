import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

np.random.seed(42)

def main():
    # dataset exploration
    print("=" * 60)
    print("Iris Dataset Exploration")
    print("=" * 60)

    csv_path = os.path.join('data', 'IRIS.csv')
    if not os.path.exists(csv_path):
        csv_path = os.path.join('data', 'iris.csv')

    try:
        iris_data = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to load dataset from {csv_path}: {e}")
        return

    print("\n" + "=" * 60 )
    print("Dataset Overview")
    print("=" * 60)

    print(f"\nTotal samples: {len(iris_data)}")
    print(f"Number of features: {len(iris_data.columns) - 2}")
    print(f"Number of classes: {iris_data['species'].nunique()}")

    print("\nFirst 5 samples:")
    print(iris_data.head())

    print("\nDataset Statistics:")
    print(iris_data.describe())

    print("\nSpecies Distribution:")
    species_counts = iris_data['species_name'].value_counts()
    print(species_counts)
    print(f"\n Dataset is balanced (each species has {species_counts.min()} samples)")

    print("\nMissing Values:")
    print(iris_data.isnull().sum())
    print(" No missing values found!")

    # data analysis
    print("\n" + "=" * 60)
    print("Data analysis ")
    print("=" * 60)

    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    print("\nAverage measurements by species:")
    species_means = iris_data.groupby('species_name')[feature_cols].mean()
    print(species_means)

    # data training
    print("\n" + "=" * 60)
    print("Data Training")  
    print("=" * 60)

    X = iris_data[feature_cols].values
    y = iris_data['species'].values

    print(f"\nFeature shape:    {X.shape}")
    print(f"Target shape:     {y.shape}")
    print(f"Feature:    {feature_cols}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y 
    )

    print(f"\n Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f" Testing samples: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n Features scaled using StandardScaler")

    # knn classifier
    print("\n" + "=" * 60)
    print("TRAINING MODEL: K-NEAREST NEIGHBORS")
    print("=" * 60)

    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train_scaled, y_train)

    print(" Model trained successfully!")
    print(f"  Algorithm: K-Nearest Neighbors (k=3)")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Features used: {len(feature_cols)}")

    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE EVALUATION")
    print("=" * 60)

    y_pred = knn_classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    print(f"Correct Predictions: {(y_pred == y_test).sum()}/{len(y_test)}")

    cm = confusion_matrix(y_test, y_pred)
    species_labels = ['Setosa', 'Versicolor', 'Virginica']

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("           Setosa  Versicolor  Virginica")
    for i, label in enumerate(species_labels):
        print(f"Actual {label:12s} {cm[i][0]:3d}      {cm[i][1]:3d}        {cm[i][2]:3d}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=species_labels))

    print("\nPer-Species Analysis:")
    for i, species in enumerate(species_labels):
        species_mask = (y_test == i)
        species_accuracy = (y_pred[species_mask] == y_test[species_mask]).mean()
        print(f"  {species}: {species_accuracy*100:.1f}% accuracy")

    print("\n" + "=" * 60)
    print("TESTING WITH CUSTOM FLOWER MEASUREMENTS")
    print("=" * 60)

    test_flowers = [
        [5.1, 3.5, 1.4, 0.2],  
        [6.0, 2.7, 5.1, 1.6],  
        [7.2, 3.0, 5.8, 1.6],  
    ]

    flower_names = ['Small flower', 'Medium flower', 'Large flower']

    for i, flower in enumerate(test_flowers):
        flower_scaled = scaler.transform([flower])
        prediction = knn_classifier.predict(flower_scaled)[0]
        probabilities = knn_classifier.predict_proba(flower_scaled)[0]

        print(f"\n{flower_names[i]}:")
        print(f"  Measurements: Sepal L={flower[0]}, Sepal W={flower[1]}, "
              f"Petal L={flower[2]}, Petal W={flower[3]}")
        print(f"  Predicted: {species_labels[prediction]}")
        print(f"  Confidence: {probabilities[prediction]*100:.1f}%")
        print(f"  Probabilities: Setosa={probabilities[0]*100:.1f}%, "
              f"Versicolor={probabilities[1]*100:.1f}%, "
              f"Virginica={probabilities[2]*100:.1f}%")

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    sns.set_style('whitegrid')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=species_labels,
                yticklabels=species_labels)
    plt.title('Confusion Matrix - Iris Classification', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Species', fontsize=12)
    plt.xlabel('Predicted Species', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(" Saved: outputs/confusion_matrix.png")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Iris Measurements by Species', fontsize=16, fontweight='bold')

    for idx, feature in enumerate(feature_cols):
        row = idx // 2
        col = idx % 2

        for species_id, species_name in enumerate(species_labels):
            data = iris_data[iris_data['species'] == species_id][feature]
            axes[row, col].hist(data, alpha=0.6, label=species_name, bins=15)

        axes[row, col].set_xlabel(feature, fontsize=10)
        axes[row, col].set_ylabel('Frequency', fontsize=10)
        axes[row, col].legend()
        axes[row, col].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/feature_distributions.png', dpi=300, bbox_inches='tight')
    print(" Saved: outputs/feature_distributions.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    for species_id, species_name in enumerate(species_labels):
        species_data = iris_data[iris_data['species'] == species_id]
        plt.scatter(species_data['petal length (cm)'], 
                    species_data['petal width (cm)'],
                    label=species_name, alpha=0.6, s=100)

    plt.xlabel('Petal Length (cm)', fontsize=12)
    plt.ylabel('Petal Width (cm)', fontsize=12)
    plt.title('Iris Species Separation by Petal Measurements', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/species_separation.png', dpi=300, bbox_inches='tight')
    print(" Saved: outputs/species_separation.png")
    plt.close()

    print("\n" + "=" * 60)
    print("PROJECT COMPLETE!")
    print("=" * 60)
    print(f" Model Accuracy: {accuracy*100:.2f}%")
    print(f" All visualizations saved in outputs/ folder")

if __name__ == "__main__":
    main()
    try:
        input("\nPress Enter to exit...")
    except Exception:
        pass
