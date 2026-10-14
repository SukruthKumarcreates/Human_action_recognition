import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and combine all CSVs
all_files = glob.glob("dataset/*.csv")
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.2f}")

with open("har_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save labels too
with open("action_labels.txt", "w") as f:
    for label in sorted(y.unique()):
        f.write(label + "\n")
