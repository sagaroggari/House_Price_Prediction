import os
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "house_prices.csv"
MODEL_FILE = BASE_DIR / "models" / "house_price_model.pkl"
OUTPUT_DIR = BASE_DIR / "outputs"
HISTORY_FILE = OUTPUT_DIR / "prediction_history.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI House Price Prediction",
    page_icon="🏠",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


# =========================================================
# LOAD PROJECT FILES
# =========================================================

try:

    df = load_data()
    model = load_model()

except FileNotFoundError as e:

    st.error(f"Required file not found: {e}")

    st.info(
        "Check that data/house_prices.csv and "
        "models/house_price_model.pkl exist."
    )

    st.stop()

except Exception as e:

    st.error(f"Unable to load the project: {e}")

    st.stop()


# =========================================================
# FEATURES
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


# =========================================================
# CHECK DATASET
# =========================================================

required_columns = features + ["Price"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    st.error(
        f"Missing columns in house_prices.csv: "
        f"{missing_columns}"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏠 House Price AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "🔮 Prediction",
        "📊 Dashboard",
        "📈 Data Analysis",
        "🤖 Model Performance",
        "🔍 Explainable AI",
        "📝 Prediction History",
        "📋 Dataset"
    ]
)


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🏠 AI House Price Prediction")

st.caption(
    "Machine Learning application using "
    "Python, Pandas, Scikit-learn, Random Forest and Streamlit."
)

st.success(
    "✅ Saved Random Forest model loaded successfully!"
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.header("📊 House Price Dashboard")

    total_houses = len(df)

    average_price = df["Price"].mean()

    minimum_price = df["Price"].min()

    maximum_price = df["Price"].max()


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🏠 Total Houses",
            f"{total_houses}"
        )

    with col2:

        st.metric(
            "💰 Average Price",
            f"₹{average_price:,.0f}"
        )

    with col3:

        st.metric(
            "⬇️ Minimum Price",
            f"₹{minimum_price:,.0f}"
        )

    with col4:

        st.metric(
            "⬆️ Maximum Price",
            f"₹{maximum_price:,.0f}"
        )


    st.subheader("🏠 Property Type Summary")

    summary = (
        df.groupby("PropertyType")["Price"]
        .agg(
            ["count", "mean", "min", "max"]
        )
        .reset_index()
    )

    summary.columns = [
        "Property Type",
        "Number of Houses",
        "Average Price",
        "Minimum Price",
        "Maximum Price"
    ]

    st.dataframe(
        summary,
        use_container_width=True
    )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "📈 Data Analysis":

    st.header("📈 Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📍 Area vs Price",
            "🏠 Property Type",
            "📊 Price Distribution",
            "🔥 Correlation"
        ]
    )


    # -----------------------------------------------------
    # AREA VS PRICE
    # -----------------------------------------------------

    with tab1:

        st.subheader(
            "📍 Area vs House Price"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.scatter(
            df["Area"],
            df["Price"]
        )

        ax.set_xlabel(
            "Area (sq ft)"
        )

        ax.set_ylabel(
            "Price (₹)"
        )

        ax.set_title(
            "Area vs House Price"
        )

        ax.grid(True)

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


    # -----------------------------------------------------
    # PROPERTY TYPE
    # -----------------------------------------------------

    with tab2:

        st.subheader(
            "🏠 Average Price by Property Type"
        )

        average_property_price = (
            df.groupby("PropertyType")["Price"]
            .mean()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(
            average_property_price.index,
            average_property_price.values
        )

        ax.set_xlabel(
            "Property Type"
        )

        ax.set_ylabel(
            "Average Price (₹)"
        )

        ax.set_title(
            "Average Price by Property Type"
        )

        ax.tick_params(
            axis="x",
            rotation=15
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


    # -----------------------------------------------------
    # PRICE DISTRIBUTION
    # -----------------------------------------------------

    with tab3:

        st.subheader(
            "📊 House Price Distribution"
        )

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.hist(
            df["Price"],
            bins=8,
            edgecolor="black"
        )

        ax.set_xlabel(
            "House Price (₹)"
        )

        ax.set_ylabel(
            "Number of Houses"
        )

        ax.set_title(
            "Distribution of House Prices"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


    # -----------------------------------------------------
    # CORRELATION
    # -----------------------------------------------------

    with tab4:

        st.subheader(
            "🔥 Numerical Feature Correlation"
        )

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

        fig, ax = plt.subplots(
            figsize=(9, 6)
        )

        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.header("🤖 Model Performance")

    X = df[features]

    y = df["Price"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


    # Predictions
    y_pred = model.predict(X_test)


    # Metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )


    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "MAE",
            f"₹{mae:,.0f}"
        )

    with col2:

        st.metric(
            "RMSE",
            f"₹{rmse:,.0f}"
        )

    with col3:

        st.metric(
            "R² Score",
            f"{r2:.4f}"
        )


    st.subheader(
        "Actual vs Predicted Price"
    )


    comparison = pd.DataFrame(
        {
            "Actual Price": y_test.values,
            "Predicted Price": y_pred
        }
    )

    comparison = comparison.sort_values(
        "Actual Price"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        comparison["Actual Price"].values,
        marker="o",
        label="Actual Price"
    )

    ax.plot(
        comparison["Predicted Price"].values,
        marker="o",
        label="Predicted Price"
    )

    ax.set_xlabel(
        "Test House"
    )

    ax.set_ylabel(
        "Price (₹)"
    )

    ax.set_title(
        "Actual vs Predicted House Prices"
    )

    ax.legend()

    ax.grid(True)

    plt.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True
    )

    plt.close(fig)


    st.subheader(
        "Actual vs Predicted Table"
    )

    st.dataframe(
        comparison,
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.header("🔮 Predict House Price")

    st.write(
        "Enter the details of a new house."
    )


    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # LEFT INPUTS
    # -----------------------------------------------------

    with col1:

        area = st.number_input(
            "Area (sq ft)",
            min_value=100.0,
            max_value=100000.0,
            value=1500.0,
            step=50.0
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=20,
            value=3,
            step=1
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        age = st.number_input(
            "House Age (years)",
            min_value=0.0,
            max_value=200.0,
            value=4.0,
            step=1.0
        )


    # -----------------------------------------------------
    # RIGHT INPUTS
    # -----------------------------------------------------

    with col2:

        parking = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

        location = st.selectbox(
            "Location",
            sorted(
                df["Location"]
                .dropna()
                .unique()
            )
        )

        furnished = st.selectbox(
            "Furnished Status",
            sorted(
                df["Furnished"]
                .dropna()
                .unique()
            )
        )

        property_type = st.selectbox(
            "Property Type",
            sorted(
                df["PropertyType"]
                .dropna()
                .unique()
            )
        )


    predict_button = st.button(
        "🔮 Predict House Price",
        use_container_width=True
    )


    if predict_button:

        # -------------------------------------------------
        # VALIDATE INPUT
        # -------------------------------------------------

        if area <= 0:

            st.error(
                "Area must be greater than 0."
            )

            st.stop()


        if bedrooms <= 0:

            st.error(
                "Bedrooms must be greater than 0."
            )

            st.stop()


        if bathrooms <= 0:

            st.error(
                "Bathrooms must be greater than 0."
            )

            st.stop()


        if age < 0:

            st.error(
                "House age cannot be negative."
            )

            st.stop()


        if parking < 0:

            st.error(
                "Parking cannot be negative."
            )

            st.stop()


        # -------------------------------------------------
        # CREATE NEW HOUSE DATA
        # -------------------------------------------------

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


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        predicted_price = model.predict(
            new_house
        )[0]


        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.subheader(
            "💰 Prediction Result"
        )

        st.metric(
            "Predicted House Price",
            f"₹{predicted_price:,.2f}"
        )


        # -------------------------------------------------
        # HOUSE DETAILS
        # -------------------------------------------------

        st.subheader(
            "🏠 Selected House Details"
        )

        st.dataframe(
            new_house,
            use_container_width=True
        )


        # -------------------------------------------------
        # SAVE PREDICTION HISTORY
        # -------------------------------------------------

        history_data = pd.DataFrame(
            [{
                "Date":
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "Area": area,

                "Bedrooms": bedrooms,

                "Bathrooms": bathrooms,

                "Age": age,

                "Parking": parking,

                "Location": location,

                "Furnished": furnished,

                "PropertyType":
                    property_type,

                "PredictedPrice":
                    round(
                        predicted_price,
                        2
                    )
            }]
        )


        if HISTORY_FILE.exists():

            history_data.to_csv(
                HISTORY_FILE,
                mode="a",
                header=False,
                index=False
            )

        else:

            history_data.to_csv(
                HISTORY_FILE,
                index=False
            )


        st.success(
            "✅ Prediction saved successfully!"
        )


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "🔍 Explainable AI":

    st.header(
        "🔍 Explainable AI"
    )

    st.write(
        "Understand which features are most important "
        "to the Random Forest model and explain an "
        "individual prediction."
    )


    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    st.subheader(
        "📊 Random Forest Feature Importance"
    )

    try:

        rf_regressor = (
            model.named_steps["regressor"]
        )

        fitted_preprocessor = (
            model.named_steps["preprocessor"]
        )

        feature_names = (
            fitted_preprocessor
            .get_feature_names_out()
        )

        importances = (
            rf_regressor
            .feature_importances_
        )

        feature_importance = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        )

        feature_importance = (
            feature_importance
            .sort_values(
                by="Importance",
                ascending=False
            )
        )


        st.dataframe(
            feature_importance,
            use_container_width=True
        )


        top_features = (
            feature_importance
            .head(12)
            .sort_values(
                "Importance"
            )
        )


        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            top_features["Feature"],
            top_features["Importance"]
        )

        ax.set_xlabel(
            "Importance"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Top Feature Importance"
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True
        )

        plt.close(fig)


        # -------------------------------------------------
        # SHAP INPUTS
        # -------------------------------------------------

        st.subheader(
            "🧠 Explain an Individual Prediction"
        )


        col1, col2 = st.columns(2)


        with col1:

            shap_area = st.number_input(
                "Area (sq ft)",
                min_value=100.0,
                value=1500.0,
                step=50.0,
                key="shap_area"
            )

            shap_bedrooms = st.number_input(
                "Bedrooms",
                min_value=1,
                max_value=20,
                value=3,
                key="shap_bedrooms"
            )

            shap_bathrooms = st.number_input(
                "Bathrooms",
                min_value=1,
                max_value=20,
                value=2,
                key="shap_bathrooms"
            )

            shap_age = st.number_input(
                "House Age",
                min_value=0.0,
                max_value=200.0,
                value=4.0,
                key="shap_age"
            )


        with col2:

            shap_parking = st.number_input(
                "Parking Spaces",
                min_value=0,
                max_value=20,
                value=1,
                key="shap_parking"
            )

            shap_location = st.selectbox(
                "Location",
                sorted(
                    df["Location"]
                    .dropna()
                    .unique()
                ),
                key="shap_location"
            )

            shap_furnished = st.selectbox(
                "Furnished Status",
                sorted(
                    df["Furnished"]
                    .dropna()
                    .unique()
                ),
                key="shap_furnished"
            )

            shap_property_type = st.selectbox(
                "Property Type",
                sorted(
                    df["PropertyType"]
                    .dropna()
                    .unique()
                ),
                key="shap_property_type"
            )


        explain_button = st.button(
            "🧠 Explain This Prediction",
            use_container_width=True
        )


        if explain_button:

            explanation_house = pd.DataFrame(
                [[
                    shap_area,
                    shap_bedrooms,
                    shap_bathrooms,
                    shap_age,
                    shap_parking,
                    shap_location,
                    shap_furnished,
                    shap_property_type
                ]],
                columns=features
            )


            explanation_prediction = (
                model.predict(
                    explanation_house
                )[0]
            )


            st.metric(
                "Predicted House Price",
                f"₹{explanation_prediction:,.2f}"
            )


            # ---------------------------------------------
            # TRANSFORM DATA
            # ---------------------------------------------

            X_background = (
                fitted_preprocessor.transform(X)
            )

            X_explanation = (
                fitted_preprocessor.transform(
                    explanation_house
                )
            )


            if hasattr(
                X_background,
                "toarray"
            ):

                X_background = (
                    X_background.toarray()
                )


            if hasattr(
                X_explanation,
                "toarray"
            ):

                X_explanation = (
                    X_explanation.toarray()
                )


            # ---------------------------------------------
            # SHAP
            # ---------------------------------------------

            explainer = shap.TreeExplainer(
                rf_regressor,
                data=X_background,
                feature_names=feature_names
            )


            shap_result = explainer(
                X_explanation
            )


            shap_result.feature_names = (
                feature_names
            )


            st.subheader(
                "🔍 Feature Contribution"
            )


            fig = plt.figure(
                figsize=(10, 7)
            )


            shap.plots.waterfall(
                shap_result[0],
                max_display=12,
                show=False
            )


            plt.tight_layout()


            st.pyplot(
                fig,
                clear_figure=True
            )


            plt.close(fig)


    except Exception as e:

        st.error(
            f"Explainable AI error: {e}"
        )


# =========================================================
# PREDICTION HISTORY
# =========================================================

elif page == "📝 Prediction History":

    st.header(
        "📝 Prediction History"
    )


    if HISTORY_FILE.exists():

        try:

            history_df = pd.read_csv(
                HISTORY_FILE
            )


            if history_df.empty:

                st.info(
                    "Prediction history is empty."
                )

            else:

                st.dataframe(
                    history_df,
                    use_container_width=True
                )


                st.metric(
                    "Total Predictions",
                    len(history_df)
                )


                csv_data = (
                    history_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )


                st.download_button(
                    "⬇️ Download Prediction History",
                    data=csv_data,
                    file_name="prediction_history.csv",
                    mime="text/csv"
                )


                if st.button(
                    "🗑️ Clear Prediction History"
                ):

                    HISTORY_FILE.unlink()

                    st.success(
                        "Prediction history cleared."
                    )

                    st.rerun()


        except Exception as e:

            st.error(
                f"Unable to read history: {e}"
            )

    else:

        st.info(
            "No prediction history available yet."
        )


# =========================================================
# DATASET
# =========================================================

elif page == "📋 Dataset":

    st.header(
        "📋 House Price Dataset"
    )

    st.write(
        f"Total records: {len(df)}"
    )


    st.dataframe(
        df,
        use_container_width=True
    )


    st.subheader(
        "Dataset Summary"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        missing_values = int(
            df.isnull()
            .sum()
            .sum()
        )

        st.metric(
            "Missing Values",
            missing_values
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI House Price Prediction System | "
    "Python + Pandas + Scikit-learn + "
    "Random Forest + Streamlit + SHAP"
)