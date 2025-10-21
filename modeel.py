import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib  # For saving & loading model
import os      # For handling file paths

# ✅ Step 1: Load Dataset
file_path = "training_a_model/cleaned_ingredients_dataset.csv"  # Corrected path
df = pd.read_csv(file_path)

# ✅ Step 2: Preprocess Data (Remove Spaces & Missing Values)
df.dropna(inplace=True)  # Remove missing values
df["Ingredient"] = df["Ingredient"].str.lower().str.strip()
df["Health Rating"] = df["Health Rating"].astype(float)  # Ensure numeric

# ✅ Step 3: Convert Ingredients to TF-IDF Features
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["Ingredient"])
y = df["Health Rating"]

# ✅ Step 4: Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 5: Train Random Forest Model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ✅ Step 6: Evaluate Model
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Mean Absolute Error: {mae:.2f}")


# ✅ Step 7: Save Model & TF-IDF Vectorizer with Error Handling
save_dir = "training_a_model"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

try:
    joblib.dump(rf, os.path.join(save_dir, "random_forest_model.pkl"))
    joblib.dump(tfidf, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
    print("✅ Model & Vectorizer Saved Successfully!")
except PermissionError:
    print("❌ Permission denied: Unable to save model files. Check folder access rights.")
except Exception as e:
    print(f"❌ Failed to save model/vectorizer: {e}")
    

# ✅ Step 8: Make Predictions on New Ingredients
new_ingredients = [
    "Corn flour", "sugar", "oat", "flour", "brown sugar",
    "palm and/or coconut oil", "salt", "sodium citrate",
    "natural and artificial flavor", "malic acid"
]
new_tfidf = tfidf.transform(new_ingredients)
predictions = rf.predict(new_tfidf)

# ✅ Print Predictions
print("\n🔹 Ingredient Predictions:")
for ingr, pred in zip(new_ingredients, predictions):
    print(f"🍏 {ingr}: {pred:.2f}")
