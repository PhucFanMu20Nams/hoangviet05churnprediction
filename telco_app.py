import streamlit as st
import pandas as pd
# import zipfile # No longer needed
# import os      # No longer needed
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier # Using this model
import numpy as np

# Set up page
st.set_page_config(page_title="Telco Churn Predictor", layout="wide") # Using wide layout
st.title("📉 Telco Customer Churn Predictor")

# --- Inject CSS for Styling ---
# This CSS block styles elements in the main area AND sets the background
st.markdown("""
<style>
/* Target the Streamlit App's main container for the background */
[data-testid="stAppViewContainer"] {
    background-color: #FFFFE0; /* LightYellow */
}

/* Style Info/Alert Boxes to have white background */
div[data-testid="stAlert"] {
    background-color: #FFFFFF !important; /* White background, !important to override theme */
    border: 1px solid #DCDCDC;       /* Add border for definition */
    border-radius: 0.25rem;          /* Match other rounding */
    color: #31333F;                  /* Ensure text color is dark */
}
/* Ensure icons inside alerts are visible */
 div[data-testid="stAlert"] svg {
    fill: #0068C9; /* Standard blue for info icon */
 }
 div[data-testid="stAlert"][data-alert-success="true"] svg {
    fill: #008000; /* Green for success icon */
 }

/* Style the Expander Header - SET TO WHITE */
div[data-testid="stExpander"] summary {
    background-color: #FFFFFF !important; /* White + !important */
    border: 1px solid #d3d3d3;
    border-radius: 0.25rem; /* Only round header corners initially */
    padding: 0.5rem 1rem;
    margin-top: 1em;
    transition: background-color 0.3s ease;
    color: #31333F; /* Ensure header text is dark */
}
div[data-testid="stExpander"] summary:hover {
    background-color: #e9ecef !important; /* Keep hover effect, add !important */
}
/* Style the Expander Content Area - SET TO WHITE (Using Adjacent Sibling Selector) */
div[data-testid="stExpander"] summary + div { /* Targets the div directly after the summary */
    background-color: #FFFFFF !important; /* White background + !important */
    padding: 1rem;             /* Add some padding inside the content area */
    border: 1px solid #d3d3d3; /* Add border matching the header */
    border-top: none;          /* Remove top border */
    border-radius: 0 0 0.25rem 0.25rem; /* Round bottom corners */
    margin-top: -1px;          /* Adjust slightly to align with header border */
    color: #31333F;            /* Ensure content text is dark */
}


/* Styles specifically targeting elements *within* the main content block */
div[data-testid="stAppViewContainer"] > section[data-testid="stBlock"] {

    /* Style the Header for Prediction Result */
    div[data-testid="stVerticalBlock"] [data-testid="stMarkdownContainer"] h3 {
        color: #465a70;
        padding-top: 1em;
        border-top: 1px solid #e0e0e0;
        margin-top: 1em;
    }

    /* Style the 'Go back' Link Button */
    a.stLinkButton {
        background-color: #EAEAEA;
        color: #31333F !important; /* Use #31333F for consistency */
        padding: 0.5em 1em;
        border-radius: 0.25rem;
        text-decoration: none;
        border: 1px solid #D3D3D3;
        display: inline-block;
        line-height: normal;
        font-weight: normal;
        margin-top: 1em;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    a.stLinkButton:hover {
        background-color: #DCDCDC;
        border-color: #BEBEBE;
        color: #31333F !important;
        text-decoration: none;
    }
    a.stLinkButton:active {
        background-color: #C0C0C0;
    }

    /* Style the Selectbox background to white */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child {
        background-color: #FFFFFF; /* White background */
        border: 1px solid #D3D3D3; /* Add a light border */
        border-radius: 0.25rem;    /* Match rounding */
        color: #31333F !important;  /* Ensure text is dark */
    }
    /* Ensure selected value text is dark */
    div[data-testid="stSelectbox"] span {
        color: #31333F;
    }
    /* Style the dropdown arrow */
    div[data-testid="stSelectbox"] svg {
        fill: #31333F; /* Make arrow dark */
    }

    /* Restore Help (?) Icon Color */
    div[data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] svg {
        fill: #808080; /* Set icon color to grey */
    }

     /* Primary Button styles (Optional - keep commented out) */
    /*
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #5F9EA0;
        color: white; border: none; transition: background-color 0.3s ease;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button[kind="primary"]:hover {
        background-color: #538d8e; border: none;
    }
    div[data-testid="stVerticalBlock"] div[data-testid="stButton"] > button[kind="primary"]:active {
        background-color: #487a7b; border: none;
    }
    */

} /* End of section[data-testid="stBlock"] specific styles */
</style>
""", unsafe_allow_html=True)
# End of CSS block


# --- Initialize Session State Flags ---
if 'predict_pressed' not in st.session_state:
    st.session_state.predict_pressed = False
if 'prediction_confirmed' not in st.session_state:
    st.session_state.prediction_confirmed = False
# -------------------------------------

# --- Data Loading ---
def load_data():
    """Loads, cleans, and returns the Telco churn dataset."""
    csv_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    try:
        df = pd.read_csv(csv_path)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(subset=['TotalCharges'], inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file '{csv_path}' was not found.")
        st.info("Place 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the script's folder.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        st.stop()
# --------------------

# --- Model Training (Using RandomForestClassifier) ---
@st.cache_data
def train_model(_df):
    """Trains the RandomForest model using original logic and returns components."""
    # Note: st.info calls will appear at the top before columns render
    st.info("⚙️ Training model (this happens once)...") # Should have white background now
    df_model = _df.drop(columns=['customerID'])
    categorical_cols = df_model.select_dtypes(include='object').columns.drop('Churn')
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    if 'Churn' in df_model.columns:
        le_churn = LabelEncoder()
        df_model['Churn'] = le_churn.fit_transform(df_model['Churn'])
        label_encoders['Churn'] = le_churn
    else:
        st.error("Target 'Churn' column not found.")
        st.stop()

    try:
        X = df_model.drop(columns=['Churn'])
        y = df_model['Churn']
    except KeyError:
        st.error("Could not separate features/target.")
        st.stop()

    num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    if not all(col in X.columns for col in num_cols_to_scale):
        st.error(f"Numeric columns for scaling not found.")
        st.stop()

    scaler = StandardScaler()
    X[num_cols_to_scale] = scaler.fit_transform(X[num_cols_to_scale])

    # --- Using RandomForestClassifier ---
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    # ------------------------------------
    model.fit(X, y)
    st.info("✅ Model training complete.") # Should have white background now
    return model, label_encoders, scaler, X.columns.tolist()
# --------------------------

# --- Prepare Input Function (Definition only, no change) ---
def prepare_input(user_input_series, label_encoders_map, scaler_obj, feature_order):
    input_df = pd.DataFrame([user_input_series])
    for col, le in label_encoders_map.items():
        if col in input_df.columns and col not in ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn']:
             try: input_df[col] = le.transform(input_df[col])
             except ValueError:
                  st.warning(f"Unseen value '{input_df[col].iloc[0]}' in '{col}'. Using default.")
                  default_val = 'No'; input_df[col] = le.transform([default_val])[0] if default_val in le.classes_ else -1
             except Exception as e: st.error(f"Encoding error '{col}': {e}"); input_df[col] = -1
    num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    try:
        input_df['TotalCharges'] = input_df['TotalCharges'].astype(float)
        input_df[num_cols_to_scale] = scaler_obj.transform(input_df[num_cols_to_scale])
    except Exception as e: st.error(f"Scaling error: {e}"); return None
    try: input_df = input_df[feature_order]
    except KeyError as e: st.error(f"Feature order error: {e}"); return None
    return input_df.values
# ----------------------------------------


# --- Main Application Logic ---
df = load_data()

if df is not None:
    model, label_encoders, scaler, feature_names = train_model(df.copy())

    # --- Define Default Values for Reset ---
    try:
        default_values = {
            'gender_input': label_encoders['gender'].classes_[0],
            'senior_input': "Yes", 'partner_input': label_encoders['Partner'].classes_[0],
            'dependents_input': label_encoders['Dependents'].classes_[0], 'tenure_input': 12,
            'phone_input': label_encoders['PhoneService'].classes_[0], 'multiple_input': label_encoders['MultipleLines'].classes_[0],
            'internet_input': label_encoders['InternetService'].classes_[0], 'onlinesec_input': label_encoders['OnlineSecurity'].classes_[0],
            'onlinebackup_input': label_encoders['OnlineBackup'].classes_[0], 'protection_input': label_encoders['DeviceProtection'].classes_[0],
            'tech_input': label_encoders['TechSupport'].classes_[0], 'tv_input': label_encoders['StreamingTV'].classes_[0],
            'movies_input': label_encoders['StreamingMovies'].classes_[0], 'contract_input': label_encoders['Contract'].classes_[0],
            'paperless_input': label_encoders['PaperlessBilling'].classes_[0], 'payment_input': label_encoders['PaymentMethod'].classes_[0],
            'charges_input': 65.0
        }
    except Exception as e:
        st.error(f"Error defining default values: {e}")
        st.stop()
    # ----------------------------------------

    # --- Updated Reset Callback ---
    def reset_widgets():
        """Callback to reset input widgets AND control flags."""
        for k, v in default_values.items():
            if k in st.session_state:
                st.session_state[k] = v
        # Reset control flags
        st.session_state.predict_pressed = False
        st.session_state.prediction_confirmed = False
    # ---------------------------

    # --- Define Columns for Layout ---
    col_main, col_input = st.columns([2, 1]) # Left column 2x width of right
    # ---------------------------------

    # --- Input Section (Right Column) ---
    with col_input:
        st.header("📝 Customer Input")
        is_disabled_state = st.session_state.get('prediction_confirmed', False)

        if is_disabled_state:
            st.caption("Inputs locked. Press 'Reset Inputs' to change.")
        else:
            st.caption("Adjust inputs, then click 'Predict Churn'.")

        # --- Get user inputs directly within this column ---
        gender = st.selectbox("Gender", options=label_encoders['gender'].classes_, key='gender_input', help="Customer's gender", disabled=is_disabled_state)
        senior_selection = st.selectbox("Senior Citizen", ["Yes", "No"], key='senior_input', help="Is the customer a senior citizen (65+)?", disabled=is_disabled_state)
        senior_map = {"Yes": 1, "No": 0}; senior = senior_map[senior_selection]
        partner = st.selectbox("Has Partner?", options=label_encoders['Partner'].classes_, key='partner_input', help="Does the customer have a partner?", disabled=is_disabled_state)
        dependents = st.selectbox("Has Dependents?", options=label_encoders['Dependents'].classes_, key='dependents_input', help="Does the customer have dependents?", disabled=is_disabled_state)
        tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12, key='tenure_input', help="Number of months the customer has stayed", disabled=is_disabled_state)
        phone = st.selectbox("Phone Service", options=label_encoders['PhoneService'].classes_, key='phone_input', help="Does the customer have phone service?", disabled=is_disabled_state)
        multiple = st.selectbox("Multiple Lines", options=label_encoders['MultipleLines'].classes_, key='multiple_input', help="Does the customer have multiple phone lines?", disabled=is_disabled_state)
        internet = st.selectbox("Internet Service", options=label_encoders['InternetService'].classes_, key='internet_input', help="Type of internet service", disabled=is_disabled_state)
        online_sec = st.selectbox("Online Security", options=label_encoders['OnlineSecurity'].classes_, key='onlinesec_input', help="Does the customer have online security service?", disabled=is_disabled_state)
        online_backup = st.selectbox("Online Backup", options=label_encoders['OnlineBackup'].classes_, key='onlinebackup_input', help="Does the customer have online backup service?", disabled=is_disabled_state)
        protection = st.selectbox("Device Protection", options=label_encoders['DeviceProtection'].classes_, key='protection_input', help="Does the customer have device protection service?", disabled=is_disabled_state)
        tech = st.selectbox("Tech Support", options=label_encoders['TechSupport'].classes_, key='tech_input', help="Does the customer have tech support service?", disabled=is_disabled_state)
        tv = st.selectbox("Streaming TV", options=label_encoders['StreamingTV'].classes_, key='tv_input', help="Does the customer stream TV?", disabled=is_disabled_state)
        movies = st.selectbox("Streaming Movies", options=label_encoders['StreamingMovies'].classes_, key='movies_input', help="Does the customer stream movies?", disabled=is_disabled_state)
        contract = st.selectbox("Contract", options=label_encoders['Contract'].classes_, key='contract_input', help="Customer's contract term", disabled=is_disabled_state)
        paperless = st.selectbox("Paperless Billing", options=label_encoders['PaperlessBilling'].classes_, key='paperless_input', help="Does the customer use paperless billing?", disabled=is_disabled_state)
        payment = st.selectbox("Payment Method", options=label_encoders['PaymentMethod'].classes_, key='payment_input', help="Customer's payment method", disabled=is_disabled_state)
        charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=0.05, key='charges_input', help="Customer's current monthly charge", disabled=is_disabled_state)

        # --- Create input series from collected values ---
        total_charges_calc = float(charges * tenure)
        input_dict = {'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,'OnlineSecurity': online_sec, 'OnlineBackup': online_backup, 'DeviceProtection': protection,'TechSupport': tech, 'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': charges,'TotalCharges': total_charges_calc }
        try:
            # This series holds the current inputs from the right column
            user_input_series = pd.Series({col: input_dict[col] for col in feature_names})
        except KeyError as e:
            st.error(f"Input key error during dict creation: {e}")
            # Avoid stopping the whole app if possible, maybe disable prediction
            user_input_series = None # Indicate error
        # -------------------------------------------------

    # --- Main Interaction Area (Left Column) ---
    with col_main:
        # Display Predict button IF no prediction is confirmed yet
        if not st.session_state.get('prediction_confirmed', False):
            if st.button("➡️ Predict Churn", key="predict_button_main", help="Click after setting all inputs."):
                st.session_state.predict_pressed = True
                st.session_state.prediction_confirmed = False
                st.rerun()

        # --- Confirmation Step ---
        if st.session_state.get('predict_pressed', False) and not st.session_state.get('prediction_confirmed', False):
            st.warning("**Are you sure with these category inputs?**") # st.warning has default style
            sub_col_confirm, sub_col_cancel = st.columns(2) # Use sub-columns within col_main
            with sub_col_confirm:
                # Primary button uses theme color (via config.toml) or CSS if defined
                if st.button("✅ Yes, Predict!", key="confirm_yes"):
                    st.session_state.prediction_confirmed = True
                    st.session_state.predict_pressed = False
                    st.rerun()
            with sub_col_cancel:
                 # Secondary button uses default theme style
                if st.button("❌ No, Change Inputs", key="confirm_no"):
                    st.session_state.predict_pressed = False
                    st.session_state.prediction_confirmed = False
                    st.rerun()
        # ------------------------

        # --- Predict and Display Results ---
        if st.session_state.get('prediction_confirmed', False):
            # Ensure model and inputs are ready before proceeding
            if 'model' in locals() and user_input_series is not None:
                prepared_input = prepare_input(user_input_series, label_encoders, scaler, feature_names)
                if prepared_input is not None:
                    proba = model.predict_proba(prepared_input)[0][1] # Using the RandomForest model

                    # CSS styles the h3 tag below
                    st.markdown("### 🔮 Prediction Result")
                    st.metric(label="Churn Probability", value=f"{proba:.2%}", delta=None)

                    if proba > 0.5:
                        st.error("⚠️ High Risk of Churn") # st.error has default style
                    else:
                        st.success("✅ Low Risk of Churn") # st.success has default style

                    # Reset button uses default secondary style
                    st.button("🔄 Reset Inputs", key="reset_button_after_predict", on_click=reset_widgets, help="Click to reset inputs, hide prediction, and unlock inputs.")
                else:
                     st.error("Could not generate prediction due to input preparation error.")
            elif user_input_series is None:
                 st.error("Could not generate prediction due to input gathering error.")
            else: # model not in locals
                 st.error("Model components not loaded correctly. Cannot predict.")
        # --- End of Conditional Display ---

        # --- Footer and Extended Information (Expanded Content) ---
        # CSS adds margin/padding, no need for st.markdown("---") here
        with st.expander("ℹ️ Detailed Information: App Functionality & Random Forest Model", expanded=False): # Header/Content now white
            st.markdown(
                """
                ### Welcome to the Telco Customer Churn Predictor!

                This application leverages machine learning to estimate the probability of a telecommunications customer **churning** – meaning, deciding to stop using the company's services. Understanding and predicting churn is crucial for businesses like Telcos, as retaining existing customers is often more cost-effective than acquiring new ones. Early identification of customers at high risk allows the company to potentially intervene with targeted offers, improved support, or other retention strategies. This tool simulates that prediction process.

                **How it Works:**
                You provide details about a hypothetical or real customer using the input fields on the right. These details cover several categories:
                * **Demographics:** Gender, whether they are a senior citizen, if they have partners or dependents. These factors can sometimes correlate with lifestyle changes or different service needs that might influence churn.
                * **Account Tenure & Contract:** `Tenure` (how many months they've been a customer) is often a strong indicator of loyalty. `Contract` type (Month-to-month, One year, Two year) is critical; shorter contracts typically have higher churn rates. `PaymentMethod` and `PaperlessBilling` preferences also provide behavioral insights.
                * **Service Usage:** Details about core services like `PhoneService` and `InternetService` (DSL, Fiber Optic, or None), along with add-ons like `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies`. The combination and type of services used can indicate customer engagement level and potential dependencies on the provider.
                * **Billing Information:** `MonthlyCharges` and `TotalCharges` reflect the financial aspect of the customer relationship. Significant changes or high relative costs can be churn drivers. *(Note: In this app, TotalCharges for prediction is estimated dynamically based on the input Monthly Charges and Tenure for consistency within the prediction request).*

                Once you confirm the inputs, the application feeds this information into a pre-trained machine learning model.

                **The Model Used:**
                The predictive engine behind this app is a **Random Forest Classifier**. This specific type of model was chosen for its generally high accuracy in classification tasks, its ability to handle a mix of numerical and categorical features (after encoding), and its relative robustness against overfitting compared to single decision trees. The model was trained using the well-known **"Telco Customer Churn" dataset**, often sourced from platforms like Kaggle, which contains information on thousands of past customers and whether they ultimately churned or not. The goal of the training process was for the model to learn the complex patterns and relationships between customer attributes and their likelihood of churning.

                **Output:**
                The primary output is the **Churn Probability** – a percentage indicating the model's estimated likelihood that a customer with the given characteristics will churn. Based on this probability (typically using a 50% threshold, but this can be adjusted), the app provides a simple risk assessment: Low Risk or High Risk.

                **Data Requirement:**
                For the underlying model training (which happens only once when the app starts or the cache is cleared), the application needs access to the `WA_Fn-UseC_-Telco-Customer-Churn.csv` data file located in the same directory as the Python script.
                """
            )

            st.markdown("---") # Separator within the expander

            st.subheader("Deep Dive: The Random Forest Algorithm")
            st.markdown(
                 """
                **Core Concept: Ensemble Learning with Decision Trees**

                At its heart, a Random Forest is about the "wisdom of the crowd." Instead of relying on a single predictive model, it builds and consults many models—specifically, **Decision Trees**—and combines their outputs.

                **1. Decision Trees Explained:**
                A decision tree is like an automated version of the game "20 Questions." It asks a series of questions about the data to arrive at a conclusion.
                * **Structure:** It has a tree-like structure with a starting 'root' node representing the entire dataset.
                * **Nodes & Splits:** Each internal node represents a "test" on a specific feature (e.g., "Is Tenure < 12.5 months?"). Based on the answer, the data follows a 'branch' to the next node. The algorithm chooses the feature and split point (or category) that best divides the data into more homogeneous groups regarding the target variable (churn or not churn). This "best split" is often determined by minimizing metrics like **Gini Impurity** (measuring the likelihood of misclassifying a randomly chosen element if labeled according to the node's class distribution) or maximizing **Information Gain** (based on entropy, measuring the reduction in uncertainty).
                * **Leaves:** The tree grows until it reach 'leaf' nodes, which represent the final predicted outcome (e.g., "Churn" or "Not Churn") for data points ending up in that leaf. Usually, the prediction is the majority class of the training samples in that leaf.
                * **Challenge:** Single decision trees, especially deep ones, are prone to **overfitting**. They can learn the training data perfectly, including its noise and specific quirks, but then fail to generalize well to new, unseen data.

                **2. Building the "Forest": Randomness is Key**
                Random Forest counteracts the overfitting tendency of individual trees by introducing randomness in two crucial ways while building many trees:

                * **Bootstrap Aggregating (Bagging):** For each tree built in the forest (e.g., 100 trees), a random subset of the *original training data* is selected *with replacement*. This means a particular customer's data might be used multiple times in the training set for one tree, while being completely absent from another tree's training set. This ensures each tree learns from a slightly different perspective of the data. Data points *not* selected for a specific tree's bootstrap sample are called "Out-Of-Bag" (OOB) samples for that tree.
                * **Feature Randomness (Subspace Sampling):** When deciding on the best split at any given node within a tree, the algorithm doesn't evaluate *all* possible features. Instead, it selects a *random subset* of features (e.g., only 5 out of 20 features) and finds the best split *only among those selected features*. This prevents strong predictor features from dominating all trees and forces the model to explore different combinations of features, leading to more diverse and less correlated trees. The number of features to consider at each split (`max_features`) is an important hyperparameter.

                **3. Making Predictions: Voting or Averaging**
                Once the forest (e.g., 100 diverse trees) is built:
                * **Classification (like this app):** A new customer's data is fed through *every tree* in the forest. Each tree makes its own prediction ('Churn' or 'Not Churn'). The Random Forest then counts the votes, and the prediction that gets the **majority vote** is the final output. The proportion of trees voting for 'Churn' is used as the churn probability.
                * **Regression:** For predicting numerical values, the predictions from all trees are typically averaged.

                **4. Benefits & Drawbacks:**
                * **Advantages:** Generally high prediction accuracy; robust to outliers and non-linear data; handles high dimensions and large datasets; reduces overfitting significantly compared to single trees; requires less explicit feature scaling than some other methods (like SVM or logistic regression); provides useful **feature importance** measures (indicating which inputs were most influential in the model's decisions, often based on Gini impurity reduction or permutation accuracy). It can also estimate its own generalization error internally using the OOB samples.
                * **Disadvantages:** Can be computationally intensive and require more memory (storing many trees); results in a "black box" model that is much less interpretable than a single decision tree (it's hard to trace the exact path leading to a prediction); might require tuning hyperparameters like the number of trees (`n_estimators`), `max_features`, maximum tree depth (`max_depth`), minimum samples per leaf (`min_samples_leaf`), etc., for optimal performance.
                """
            )

            st.subheader("Random Forest Explained Simply (The Very Smart Guessing Game)")
            st.markdown(
                 """
                Imagine you want to guess if a friend will like a new movie (`Churn` vs. `Not Churn`). Asking just one friend might give you a biased answer based on their unique tastes (like a single Decision Tree expert).

                A Random Forest is like playing a **super-smart guessing game with a huge team of friends**:

                1.  **Gather Your Team (Many Trees):** You get a big team together, maybe 100 friends (these are your 'Decision Trees').

                2.  **Give Them Different Clues:** You don't tell every friend *everything* about the new movie and the friend whose preference you're predicting.
                    * **Random Movie History (Bagging):** You give each friend on the team a *random list* of movies the target friend has liked or disliked in the past. Because it's random *with replacement*, some movies might appear on multiple lists, and some lists might miss certain movies entirely. So, each friend studies a slightly different history.
                    * **Random Focus Points (Feature Randomness):** When each friend is deciding *why* the target friend might like or dislike the *new* movie, you tell them to only consider a *few random factors* at each step. For example, Friend A might only be allowed to think about "Genre" and "Lead Actor" at one point. Friend B might only get to consider "Director" and "Running Time". Friend C looks at "Genre" and "Has Action Scenes?". This prevents everyone from just focusing on the most obvious factor (like "Is it a superhero movie?") and forces them to find other patterns.

                3.  **Ask for Their Guess:** Now you describe the *new* movie to *all 100 friends* on your team. Based on the unique movie history they studied and the random factors they were allowed to focus on, each friend makes their own independent guess: "Like!" (`Not Churn`) or "Dislike!" (`Churn`).

                4.  **Count the Votes!** You collect all 100 guesses. Maybe 80 friends guess "Like!" and 20 guess "Dislike!".

                5.  **The Team's Final Answer:** The Random Forest goes with the majority. Since most friends guessed "Like!", the final prediction is "Like!" (`Not Churn`), and you can even say the confidence (probability) is pretty high (around 80%).

                **Why is this better?**
                * **Reduces Weird Opinions:** If one friend has really unusual taste (an outlier) or fixates on one odd reason (overfitting), their vote gets drowned out by the majority.
                * **Catches More Angles:** Because friends focus on different random factors, the team is more likely to catch various reasons why someone might like or dislike the movie. One friend might notice the genre match, another the director, another the actor combination.
                * **More Reliable:** The final answer based on the "wisdom of the diverse crowd" is usually much more accurate and reliable than relying on just one person's guess. That's the magic of the Random Forest! 🤔🌳🌳🌳➡️🗳️➡️🏆
                """
            )
        # --------------

        # --- Add Redirect Link Button Below Expander ---
        # CSS adds margin, no need for st.markdown("---")
        st.link_button("Go back", "https://hoangviet05.com/", help="Visit HoangViet05.com (opens in new tab)")
        # ---------------------------------------------

# --- Fallback Error Message ---
else: # df is None
    st.error("⛔ Failed to load data. Application cannot proceed.")
# -----------------------------