import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. CREATE REQUIRED FOLDERS
# =========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================================================
# 2. LOAD DATASET
# =========================================================

df = pd.read_csv("data/house_prices.csv")

print("\n========== HOUSE PRICE DATA ==========")
print(df.to_string(index=False))


# =========================================================
# 3. BASIC DATA ANALYSIS
# =========================================================

print("\n========== DATASET INFORMATION ==========")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== STATISTICS ==========")
print(df.describe())


# =========================================================
# 4. INPUT FEATURES
# =========================================================

features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "Parking",
    "Location",
    "Furnished",
    "PropertyType"
]

X = df[features]


# =========================================================
# 5. OUTPUT / TARGET
# =========================================================

y = df["Price"]


# =========================================================
# 6. CATEGORICAL FEATURES
# =========================================================

categorical_features = [
    "Location",
    "Furnished",
    "PropertyType"
]


# =========================================================
# 7. PREPROCESSING
# =========================================================

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )


# =========================================================
# 8. LINEAR REGRESSION MODEL
# =========================================================

linear_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# =========================================================
# 9. RANDOM FOREST MODEL
# =========================================================

random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            create_preprocessor()
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# =========================================================
# 10. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records :", len(X_test))


# =========================================================
# 11. TRAIN BOTH MODELS
# =========================================================

linear_model.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)

print("\n========== AI MODELS ==========")
print("Linear Regression trained successfully!")
print("Random Forest trained successfully!")


# =========================================================
# 12. SAVE TRAINED RANDOM FOREST MODEL
# =========================================================

model_path = "models/house_price_model.pkl"

joblib.dump(
    random_forest_model,
    model_path
)

print("Random Forest model saved successfully!")
print("Model location:", model_path)


# =========================================================
# 13. PREDICT TEST DATA
# =========================================================

y_pred_lr = linear_model.predict(X_test)

y_pred_rf = random_forest_model.predict(X_test)


# =========================================================
# 14. LINEAR REGRESSION EVALUATION
# =========================================================

mae_lr = mean_absolute_error(
    y_test,
    y_pred_lr
)

rmse_lr = mean_squared_error(
    y_test,
    y_pred_lr
) ** 0.5

r2_lr = r2_score(
    y_test,
    y_pred_lr
)


# =========================================================
# 15. RANDOM FOREST EVALUATION
# =========================================================

mae_rf = mean_absolute_error(
    y_test,
    y_pred_rf
)

rmse_rf = mean_squared_error(
    y_test,
    y_pred_rf
) ** 0.5

r2_rf = r2_score(
    y_test,
    y_pred_rf
)


# =========================================================
# 16. MODEL COMPARISON
# =========================================================

print("\n========== MODEL COMPARISON ==========")

print("\nLinear Regression:")
print(f"MAE  : ₹{mae_lr:,.2f}")
print(f"RMSE : ₹{rmse_lr:,.2f}")
print(f"R²   : {r2_lr:.4f}")

print("\nRandom Forest:")
print(f"MAE  : ₹{mae_rf:,.2f}")
print(f"RMSE : ₹{rmse_rf:,.2f}")
print(f"R²   : {r2_rf:.4f}")


# =========================================================
# 17. ACTUAL VS PREDICTED TABLE
# =========================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Linear Regression": y_pred_lr,
    "Random Forest": y_pred_rf
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    "outputs/model_predictions.csv",
    index=False
)


# =========================================================
# 18. USER INPUT
# =========================================================

print("\n========== NEW HOUSE DETAILS ==========")

area = float(
    input("Enter area in sq ft: ")
)

bedrooms = int(
    input("Enter number of bedrooms: ")
)

bathrooms = int(
    input("Enter number of bathrooms: ")
)

age = float(
    input("Enter house age in years: ")
)

parking = int(
    input("Enter number of parking spaces: ")
)

location = input(
    "Enter location: "
).strip()

furnished = input(
    "Enter furnished status: "
).strip()

property_type = input(
    "Enter property type: "
).strip()


# =========================================================
# 19. VALIDATE INPUT
# =========================================================

if area <= 0:
    raise ValueError("Area must be greater than 0.")

if bedrooms <= 0:
    raise ValueError("Bedrooms must be greater than 0.")

if bathrooms <= 0:
    raise ValueError("Bathrooms must be greater than 0.")

if age < 0:
    raise ValueError("House age cannot be negative.")

if parking < 0:
    raise ValueError("Parking cannot be negative.")


# =========================================================
# 20. CREATE NEW HOUSE DATA
# =========================================================

new_house = pd.DataFrame(
    [[
        area,
        bedrooms,
        bathrooms,
        age,
        parking,
        location,
        furnished,
        property_type
    ]],
    columns=features
)


# =========================================================
# 21. PREDICT NEW HOUSE PRICE
# =========================================================

predicted_lr = linear_model.predict(
    new_house
)

predicted_rf = random_forest_model.predict(
    new_house
)


# =========================================================
# 22. DISPLAY PREDICTION
# =========================================================

print("\n========== AI HOUSE PRICE PREDICTION ==========")

print(f"Area: {area} sq ft")
print(f"Bedrooms: {bedrooms}")
print(f"Bathrooms: {bathrooms}")
print(f"Age: {age} years")
print(f"Parking: {parking}")
print(f"Location: {location}")
print(f"Furnished: {furnished}")
print(f"Property Type: {property_type}")

print(
    f"\nLinear Regression Price: ₹{predicted_lr[0]:,.2f}"
)

print(
    f"Random Forest Price: ₹{predicted_rf[0]:,.2f}"
)


# =========================================================
# 23. HISTOGRAM
# =========================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["Price"],
    bins=8,
    edgecolor="black"
)

plt.xlabel("House Price (₹)")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")

plt.tight_layout()

plt.savefig(
    "outputs/house_price_histogram.png"
)

plt.show()


# =========================================================
# 24. BAR / COLUMN CHART
# =========================================================

average_price = (
    df.groupby("PropertyType")["Price"]
    .mean()
)

plt.figure(figsize=(10, 6))

plt.bar(
    average_price.index,
    average_price.values
)

plt.xlabel("Property Type")
plt.ylabel("Average Price (₹)")
plt.title("Average House Price by Property Type")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "outputs/property_type_average_price.png"
)

plt.show()


# =========================================================
# 25. ACTUAL VS PREDICTED LINE GRAPH
# =========================================================

comparison_sorted = comparison.sort_values(
    "Actual Price"
)

plt.figure(figsize=(10, 6))

plt.plot(
    comparison_sorted["Actual Price"].values,
    marker="o",
    label="Actual Price"
)

plt.plot(
    comparison_sorted["Linear Regression"].values,
    marker="o",
    label="Linear Regression"
)

plt.plot(
    comparison_sorted["Random Forest"].values,
    marker="o",
    label="Random Forest"
)

plt.xlabel("Test House")
plt.ylabel("Price (₹)")
plt.title("Actual vs Predicted House Prices")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.show()


# =========================================================
# 26. HEATMAP
# =========================================================

numerical_data = df[
    [
        "Area",
        "Bedrooms",
        "Bathrooms",
        "Age",
        "Parking",
        "Price"
    ]
]

correlation = numerical_data.corr()

plt.figure(figsize=(10, 6))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "outputs/correlation_heatmap.png"
)

plt.show()


# =========================================================
# 27. SCATTER PLOT
# =========================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Area"],
    df["Price"]
)

plt.xlabel("Area (sq ft)")
plt.ylabel("House Price (₹)")
plt.title("Area vs House Price")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "outputs/area_vs_price.png"
)

plt.show()


# =========================================================
# 28. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

rf_regressor = (
    random_forest_model
    .named_steps["regressor"]
)

fitted_preprocessor = (
    random_forest_model
    .named_steps["preprocessor"]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importances = (
    rf_regressor.feature_importances_
)

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print("\n========== FEATURE IMPORTANCE ==========")
print(
    feature_importance.to_string(
        index=False
    )
)

feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "outputs/feature_importance.png"
)

plt.show()


# =========================================================
# 29. PROGRAM COMPLETED
# =========================================================

print("\n========== PROJECT COMPLETED ==========")
print("Dataset loaded successfully")
print("Models trained successfully")
print("Model evaluation completed")
print("Prediction completed")
print("Graphs generated")
print("Feature importance generated")
print("Saved model:", model_path)
print("Output files saved inside the outputs folder")