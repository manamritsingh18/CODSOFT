import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# STEP 1: DATA LOADING
# ============================================================

df = pd.read_csv('data/Titanic-Dataset.csv')

print("=" * 50)
print("FIRST 5 PASSENGERS:")
print("=" * 50)
print(df.head())

print("\n" + "=" * 50)
print("DATASET OVERVIEW:")
print("=" * 50)
print(f"Total passengers: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "=" * 50)
print("MISSING VALUES:")
print("=" * 50)
print(df.isnull().sum())

print("\n" + "=" * 50)
print("SURVIVAL COUNT:")
print("=" * 50)
print(df['Survived'].value_counts())
print(f"\nSurvival rate: {df['Survived'].mean() * 100:.2f}%")

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

print("\n" + "=" * 50)
print("DROPPING UNNECESSARY COLUMNS:")
print("=" * 50)

original_columns = df.columns.tolist()
print(f"Original columns: {original_columns}")

df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

print(f"\nColumns after dropping: {df.columns.tolist()}")
print(f"Removed: {len(original_columns) - len(df.columns)} columns")

print("\n" + "=" * 50)
print("HANDLING MISSING VALUES:")
print("=" * 50)

print(f"Missing values before cleaning:")
print(df.isnull().sum())

age_median = df['Age'].median()
df['Age'] = df['Age'].fillna(age_median)
print(f"\n✓ Filled {df['Age'].isnull().sum()} missing ages with median: {age_median}")

embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)
print(f"✓ Filled missing Embarked with mode: {embarked_mode}")

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

print("\n" + "=" * 50)
print("CONVERTING TEXT TO NUMBERS")
print("=" * 50)

print("Sex values BEFORE:")
print(df['Sex'].value_counts())

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

print("\nSex values AFTER:")
print(df['Sex'].value_counts())

print("\nEmbarked values BEFORE:")
print(df['Embarked'].value_counts())

embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = pd.concat([df, embarked_dummies], axis=1)
df = df.drop('Embarked', axis=1)

print("\nNew columns created:")
print(embarked_dummies.columns.tolist())

print("\n" + "=" * 50)
print("FINAL DATASET PREVIEW:")
print("=" * 50)

print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

print("\n✓ Data is ready for machine learning")

# ============================================================
# STEP 3: MODEL BUILDING
# ============================================================

print("\n" + "=" * 50)
print("BUILDING PREDICTION MODEL")
print("=" * 50)

X = df.drop('Survived', axis=1)
y = df['Survived']

print(f"Features: {X.columns.tolist()}")
print(f"Target: Survived (0=Died, 1=Survived)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✓ Training set: {len(X_train)} passengers ({len(X_train)/len(df)*100:.1f}%)")
print(f"✓ Testing set: {len(X_test)} passengers ({len(X_test)/len(df)*100:.1f}%)")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("\n✓ Model trained successfully!")

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# ============================================================
# STEP 4: MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
print(f"Correct Predictions: {(y_pred == y_test).sum()}/{len(y_test)}")

print("\nConfusion Matrix:")
print("                Predicted")
print("              Died  Survived")
print(f"Actual Died    {cm[0][0]:3d}     {cm[0][1]:3d}")
print(f"     Survived  {cm[1][0]:3d}     {cm[1][1]:3d}")

print("\nDetailed Metrics:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# ============================================================
# STEP 5: FEATURE IMPORTANCE ANALYSIS
# ============================================================

print("\n" + "=" * 50)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 50)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0],
    'Abs_Coefficient': np.abs(model.coef_[0])
})

coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print("\nFeatures ranked by importance:")
print(coefficients[['Feature', 'Coefficient']].to_string(index=False))

print("\nInterpretation:")
for idx, row in coefficients.head(3).iterrows():
    impact = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  • {row['Feature']}: {impact} survival chance")

# ============================================================
# STEP 6: EXAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 50)
print("EXAMPLE PREDICTIONS")
print("=" * 50)

for i in range(min(3, len(X_test))):
    example = X_test.iloc[i:i+1]
    pred = model.predict(example)[0]
    prob = model.predict_proba(example)[0][1]
    actual = y_test.iloc[i]
    
    print(f"\nPassenger {i+1}:")
    print(f"  Predicted: {'Survived' if pred==1 else 'Died'} (confidence: {prob*100:.1f}%)")
    print(f"  Actual: {'Survived' if actual==1 else 'Died'}")
    print(f"  Result: {'✓ Correct' if pred==actual else '✗ Incorrect'}")

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================

print("\n" + "=" * 50)
print("GENERATING VISUALIZATIONS")
print("=" * 50)

sns.set_style('whitegrid')

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Died', 'Survived'],
            yticklabels=['Died', 'Survived'])
plt.title('Confusion Matrix - Model Performance', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/confusion_matrix.png")
plt.close()

plt.figure(figsize=(10, 6))
colors = ['green' if x > 0 else 'red' for x in coefficients['Coefficient']]
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Feature Importance in Survival Prediction', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/feature_importance.png")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

survival_by_sex = df.groupby('Sex')['Survived'].mean()
axes[0, 0].bar(['Male', 'Female'], survival_by_sex, color=['lightblue', 'pink'])
axes[0, 0].set_title('Survival Rate by Gender')
axes[0, 0].set_ylabel('Survival Rate')
axes[0, 0].set_ylim([0, 1])

survival_by_class = df.groupby('Pclass')['Survived'].mean()
axes[0, 1].bar([1, 2, 3], survival_by_class, color=['gold', 'silver', 'brown'])
axes[0, 1].set_title('Survival Rate by Class')
axes[0, 1].set_ylabel('Survival Rate')
axes[0, 1].set_xlabel('Passenger Class')
axes[0, 1].set_ylim([0, 1])

axes[1, 0].hist([df[df['Survived']==0]['Age'], df[df['Survived']==1]['Age']], 
                bins=20, label=['Died', 'Survived'], color=['red', 'green'], alpha=0.6)
axes[1, 0].set_title('Age Distribution')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Count')
axes[1, 0].legend()

axes[1, 1].hist([df[df['Survived']==0]['Fare'], df[df['Survived']==1]['Fare']], 
                bins=20, label=['Died', 'Survived'], color=['red', 'green'], alpha=0.6)
axes[1, 1].set_title('Fare Distribution')
axes[1, 1].set_xlabel('Fare')
axes[1, 1].set_ylabel('Count')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('outputs/survival_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/survival_analysis.png")
plt.close()

print("\n" + "=" * 50)
print("PROJECT COMPLETE!")
print("=" * 50)
print(f"\n✓ Model Accuracy: {accuracy*100:.2f}%")
print(f"✓ Visualizations saved in outputs/ folder")
print(f"✓ Ready for submission")