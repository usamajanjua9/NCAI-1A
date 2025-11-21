# ==========================================================
# 🛠️ Predictive Maintenance – Detailed Streamlit App
# ----------------------------------------------------------
# This app demonstrates an end-to-end Predictive Maintenance
# pipeline:
#   1. Data Loading (synthetic or CSV)
#   2. Feature Engineering (rolling stats + lags)
#   3. Model Training (classification or regression)
#   4. Evaluation (metrics + plots)
#   5. Inference (predict on a selected sample)
#
# Almost every line is commented for teaching purposes.
# ==========================================================

# -------------------------------
# 1. IMPORT ALL REQUIRED PACKAGES
# -------------------------------

import time                              # Used to simulate training time and progress bar
from typing import List                  # For type hints
import numpy as np                       # Numerical operations (arrays, random numbers)
import pandas as pd                      # DataFrames and data handling
import matplotlib.pyplot as plt          # Plotting (matplotlib backend)

import streamlit as st                   # Streamlit for web UI

from sklearn.model_selection import train_test_split   # Train/test split
from sklearn.ensemble import (                         # Random Forest models
    RandomForestClassifier,
    RandomForestRegressor
)
from sklearn.preprocessing import StandardScaler       # Feature scaling
from sklearn.metrics import (                          # Evaluation metrics
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)

# ----------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION & TITLE
# ----------------------------------------

# Set basic page configuration: title, icon, and layout
st.set_page_config(
    page_title="Predictive Maintenance – Detailed App",
    page_icon="🛠️",
    layout="wide"
)

# Main title displayed at the top of the app
st.title("🛠️ Predictive Maintenance – End-to-End ML Dashboard")

# Small description text under the title
st.caption(
    "This app walks through data loading, feature engineering, model training, "
    "evaluation, and inference for predictive maintenance use-cases."
)

# ---------------------------------
# 3. INITIALIZE SESSION STATE (if needed)
# ---------------------------------
# Streamlit reruns the script on every interaction.
# Session state allows us to persist objects (like data, model) across runs.

if "features_df" not in st.session_state:
    st.session_state["features_df"] = None          # To store engineered features
if "model" not in st.session_state:
    st.session_state["model"] = None                # To store trained model
if "scaler" not in st.session_state:
    st.session_state["scaler"] = None               # To store fitted scaler
if "feature_names" not in st.session_state:
    st.session_state["feature_names"] = None        # To store feature names
if "X_test" not in st.session_state:
    st.session_state["X_test"] = None               # Test features
if "y_test" not in st.session_state:
    st.session_state["y_test"] = None               # Test labels

# ---------------------------------
# 4. SIDEBAR – GLOBAL CONFIGURATION
# ---------------------------------

# Add a header in the sidebar for configuration settings
st.sidebar.header("⚙️ Global Configuration")

# Radio button for selecting type of prediction task
task_choice = st.sidebar.radio(
    "Prediction Type:",
    ["Classification – Predict Failure (0/1)", "Regression – Predict RUL"],
    index=0
)

# Convert radio text to a simpler internal label
task_type = "classification" if "Classification" in task_choice else "regression"

# Checkbox: whether to use built-in synthetic data
use_synthetic = st.sidebar.checkbox("Use synthetic example dataset", value=True)

# Sidebar file uploader to allow user to upload their own CSV
uploaded_file = st.sidebar.file_uploader("Or upload your CSV file:", type=["csv"])

# Sidebar option: whether to shuffle and stratify (useful for classification)
shuffle_data = st.sidebar.checkbox("Shuffle data before splitting", value=True)

# ---------------------------------
# 5. DATA GENERATION / DATA LOADING
# ---------------------------------

# Subheader in the main area for Step 1
st.header("📥 Step 1 – Data Loading")

# Define a function to generate a synthetic predictive maintenance dataset
def generate_synthetic_data(
    n_machines: int = 10,
    n_cycles: int = 150
) -> pd.DataFrame:
    """
    Generate synthetic sensor time-series data for multiple machines.
    Includes:
      - machine_id
      - cycle (time index)
      - sensor_1, sensor_2, sensor_3
      - failure (binary)
      - RUL (remaining useful life)
    """
    # Fix random seed so results are reproducible
    np.random.seed(42)

    # List to collect row dictionaries
    rows = []

    # Loop over each machine
    for m in range(1, n_machines + 1):
        # Randomly choose a cycle at which this machine fails
        failure_cycle = np.random.randint(int(n_cycles * 0.5), n_cycles)

        # Loop over each cycle (time step)
        for c in range(1, n_cycles + 1):
            # Degradation factor increases when closer to failure
            # When c is much less than failure_cycle, factor is near 1
            # As c approaches failure_cycle, factor grows > 1
            degradation = 1.0 + max(0, c - (failure_cycle - 40)) / 50.0

            # Generate synthetic sensor measurements influenced by degradation
            sensor_1 = np.random.normal(50 * degradation, 3)
            sensor_2 = np.random.normal(30 * degradation, 5)
            sensor_3 = np.random.normal(80 * degradation, 4)

            # Binary failure label (1 after failure cycle, else 0)
            failure = 1 if c >= failure_cycle else 0

            # Remaining useful life: cycles left until failure, never negative
            rul = max(0, failure_cycle - c)

            # Append one row of data
            rows.append(
                dict(
                    machine_id=f"M{m:03d}",   # e.g., "M001", "M002"
                    cycle=c,                 # time index
                    sensor_1=sensor_1,
                    sensor_2=sensor_2,
                    sensor_3=sensor_3,
                    failure=failure,
                    RUL=rul
                )
            )

    # Convert list of dicts to a DataFrame
    return pd.DataFrame(rows)

# Decide which dataset to use based on user options
if use_synthetic:
    # If user chose synthetic, generate the synthetic dataset
    data = generate_synthetic_data()
    st.success("Using synthetic example dataset.")
elif uploaded_file is not None:
    # Otherwise, if a file is uploaded, read it as CSV
    data = pd.read_csv(uploaded_file)
    st.success("Custom CSV file loaded.")
else:
    # If neither synthetic nor uploaded data is available, show warning and stop
    st.warning("Please enable synthetic data OR upload a CSV file to continue.")
    st.stop()   # Stop app execution until data is available

# Display first few rows of the dataset
st.write("**Dataset preview (first 5 rows):**")
st.dataframe(data.head())

# Show basic dataset info: shape and column names
st.write(f"**Dataset shape:** {data.shape[0]} rows × {data.shape[1]} columns")
st.write("**Columns:**", list(data.columns))

# -------------------------------------
# 6. FEATURE ENGINEERING CONFIGURATION
# -------------------------------------

# Subheader for Step 2
st.header("🧮 Step 2 – Feature Engineering")

# We assume that:
#  - There is one ID column (machine/asset)
#  - One time column (cycle)
#  - One target column (failure or RUL)
# We let user select them; defaults are based on synthetic data names.

# All columns in the dataset
all_columns: List[str] = list(data.columns)

# Try to set default ID and time column if they exist
default_id_col = "machine_id" if "machine_id" in all_columns else all_columns[0]
default_time_col = "cycle" if "cycle" in all_columns else all_columns[1]

# Dropdown to choose ID column
id_col = st.selectbox("Select ID column (machine / asset):", all_columns, index=all_columns.index(default_id_col))

# Dropdown to choose time column
time_col = st.selectbox("Select time column (cycle / timestamp):", all_columns, index=all_columns.index(default_time_col))

# Suggest target columns depending on classification vs regression
if task_type == "classification":
    # For classification, we look for typical names like 'failure', 'label', etc.
    candidate_targets = [c for c in all_columns if c.lower() in ["failure", "label", "target", "y"]]
else:
    # For regression (RUL), we look for RUL or similar names
    candidate_targets = [c for c in all_columns if c.lower() in ["rul", "remaining_life", "lifetime", "y"]]

# If we found candidate target columns, use the first as default; else fallback to last column
default_target_col = candidate_targets[0] if candidate_targets else all_columns[-1]

# Dropdown for target column
target_col = st.selectbox("Select target column:", all_columns, index=all_columns.index(default_target_col))

# Determine candidate sensor columns:
# we exclude ID, time, and target columns (whatever remains could be sensor-like)
candidate_sensors = [c for c in all_columns if c not in [id_col, time_col, target_col]]

# Multiselect for sensor columns (pre-select all candidates)
sensor_cols = st.multiselect(
    "Select sensor / numeric feature columns:",
    candidate_sensors,
    default=candidate_sensors
)

# If user does not select any sensor column, we cannot proceed
if len(sensor_cols) == 0:
    st.error("Please select at least one sensor/feature column.")
    st.stop()

# Slider for rolling window size (for rolling mean/std)
window_size = st.slider(
    "Rolling window size (number of time steps):",
    min_value=3,
    max_value=60,
    value=20
)

# Checkbox: whether to add lag features (previous steps)
add_lags = st.checkbox("Add lag features (previous time steps)?", value=True)

# If adding lags, ask how many lag steps to create
if add_lags:
    n_lags = st.slider("Number of lag steps:", min_value=1, max_value=10, value=3)
else:
    n_lags = 0   # If not adding lags, this will be 0

# Button to trigger feature engineering
if st.button("🚀 Generate Engineered Features"):

    # Display a spinner while computing
    with st.spinner("Generating rolling statistics and lag features..."):
        # Make a copy of the original data
        df = data.copy()

        # Sort data by ID and time so rolling and lagging are correct
        df = df.sort_values(by=[id_col, time_col])

        # We will store transformed groups in a list
        feature_frames = []

        # Group by machine/asset ID to engineer features separately per machine
        for mid, group in df.groupby(id_col):
            # Work on a copy of this machine's data
            g = group.copy()

            # Ensure rows are sorted by time
            g = g.sort_values(by=time_col)

            # For each sensor column, compute rolling mean and std over the chosen window
            for s in sensor_cols:
                # Rolling object over the column 's'
                roll = g[s].rolling(window=window_size, min_periods=1)
                # Rolling mean feature
                g[f"{s}_roll_mean"] = roll.mean()
                # Rolling std feature (NaN replaced with 0)
                g[f"{s}_roll_std"] = roll.std().fillna(0)

            # If lag features are requested, create them for each sensor
            if add_lags and n_lags > 0:
                for s in sensor_cols:
                    for lag in range(1, n_lags + 1):
                        # shift(lag) calls earlier time steps
                        g[f"{s}_lag_{lag}"] = g[s].shift(lag)

            # Append engineered group to list
            feature_frames.append(g)

        # Concatenate all groups back into a full DataFrame
        feat_df = pd.concat(feature_frames, axis=0)

        # Drop any rows containing NaN values (from lagging at the beginning)
        feat_df = feat_df.dropna().reset_index(drop=True)

        # Save engineered DataFrame in session state for later use
        st.session_state["features_df"] = feat_df

        # Show success message with resulting shape
        st.success(f"Feature engineering completed! New shape: {feat_df.shape}")

        # Show a preview in an expander
        with st.expander("📊 Engineered Features Preview (first 5 rows)", expanded=False):
            st.dataframe(feat_df.head())
else:
    # If user has not generated features yet, we do not proceed further
    if st.session_state["features_df"] is None:
        st.info("Click **Generate Engineered Features** to move to model training.")
        st.stop()

# Retrieve engineered DataFrame from session state
features_df = st.session_state["features_df"]

# ---------------------------
# 7. MODEL TRAINING CONFIG
# ---------------------------

# Subheader for Step 3
st.header("🤖 Step 3 – Model Training")

# Prepare input features X and target y from engineered DataFrame
X = features_df.drop(columns=[target_col])  # All columns except target
y = features_df[target_col]                 # Target column

# Show shapes of X and y
st.write(f"**Feature matrix X shape:** {X.shape}")
st.write(f"**Target vector y shape:** {y.shape}")

# Three columns layout for training configuration
col1, col2, col3 = st.columns(3)

# Slider for test set size
with col1:
    test_ratio = st.slider("Test size (fraction):", 0.1, 0.4, 0.2, step=0.05)

# Slider for number of trees in Random Forest
with col2:
    n_estimators = st.slider("Random Forest trees:", 50, 500, 200, step=50)

# Slider for maximum depth of trees (0 means None)
with col3:
    max_depth_val = st.slider("Max depth (0 = None):", 0, 30, 0, step=1)
    max_depth = None if max_depth_val == 0 else max_depth_val

# Checkbox for standardizing features
scale_features = st.checkbox("Scale features with StandardScaler", value=True)

# Button to start training
if st.button("🧠 Train Model"):
    # Create a progress bar
    progress = st.progress(0)
    status_text = st.empty()

    # Simulate some steps with progress bar so user sees "training"
    for i in range(30):
        progress.progress(int((i + 1) / 30 * 100))  # update progress (0–100)
        status_text.text(f"Preparing data and training model... step {i + 1}/30")
        time.sleep(0.03)                            # tiny delay for effect

    # Train-test split; stratify for classification
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=42,
        shuffle=shuffle_data,
        stratify=y if task_type == "classification" else None
    )

    # Store X_test and y_test in session state so we can use them later
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    # Initialize scaler
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = None
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # For classification: use RandomForestClassifier
    if task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1  # use all CPU cores
        )
    else:
        # For regression: use RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    # Fit (train) the model on training data
    model.fit(X_train_scaled, y_train)

    # Save model, scaler, and feature names in session state
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["feature_names"] = X.columns.tolist()

    # Finish the progress bar and status
    progress.progress(100)
    status_text.text("✅ Training completed successfully!")
    time.sleep(0.5)
    status_text.empty()

else:
    # If training button not clicked and no model in state, then stop
    if st.session_state["model"] is None:
        st.info("Configure options and click **Train Model** to continue.")
        st.stop()

# Retrieve model, scaler, feature names, and test data from session state
model = st.session_state["model"]
scaler = st.session_state["scaler"]
feature_names = st.session_state["feature_names"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

# Apply scaler to X_test (if scaler exists)
if scaler is not None:
    X_test_scaled = scaler.transform(X_test)
else:
    X_test_scaled = X_test.values

# ---------------------------
# 8. EVALUATION & PLOTS
# ---------------------------

# Subheader for Step 4
st.header("📈 Step 4 – Model Evaluation")

# Make predictions on test set
y_pred = model.predict(X_test_scaled)

# If classification, show classification metrics and confusion matrix
if task_type == "classification":
    # Compute accuracy, precision, recall, F1
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Display metrics in four columns
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision", f"{prec:.3f}")
    c3.metric("Recall", f"{rec:.3f}")
    c4.metric("F1-score", f"{f1:.3f}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Create figure for confusion matrix
    fig_cm, ax_cm = plt.subplots()
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.figure.colorbar(im, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")

    # Annotate each cell with the integer count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )

    # Show confusion matrix in the app
    st.pyplot(fig_cm)

# If regression, show MAE, RMSE, R² and scatter plot of true vs predicted
else:
    # Compute regression metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Show metrics in three columns
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.3f}")
    c2.metric("RMSE", f"{rmse:.3f}")
    c3.metric("R²", f"{r2:.3f}")

    # Scatter plot of true vs predicted
    fig_scatter, ax_sc = plt.subplots()
    ax_sc.scatter(y_test, y_pred, alpha=0.5)
    ax_sc.set_xlabel("True target (e.g., RUL)")
    ax_sc.set_ylabel("Predicted target")
    ax_sc.set_title("True vs Predicted (Regression)")
    # Plot a diagonal line as ideal reference
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax_sc.plot([min_val, max_val], [min_val, max_val], "r--")
    # Show scatter plot in app
    st.pyplot(fig_scatter)

# ---------------------------
# 9. FEATURE IMPORTANCES
# ---------------------------

# Subheader for Step 5
st.header("🔍 Step 5 – Feature Importances")

# Random Forest models have attribute 'feature_importances_'
importances = model.feature_importances_

# Sort feature indices by importance (descending order)
sorted_idx = np.argsort(importances)[::-1]

# Sort feature names and importance values accordingly
sorted_feature_names = np.array(feature_names)[sorted_idx]
sorted_importances = importances[sorted_idx]

# Take top N features (e.g., 20)
top_n = min(20, len(sorted_feature_names))
top_features = sorted_feature_names[:top_n]
top_values = sorted_importances[:top_n]

# Create bar plot for feature importances
fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
ax_imp.barh(top_features[::-1], top_values[::-1])   # Reverse for nicer display
ax_imp.set_xlabel("Importance")
ax_imp.set_title("Top Feature Importances")
st.pyplot(fig_imp)

# ---------------------------
# 10. INFERENCE / WHAT-IF
# ---------------------------

# Subheader for Step 6
st.header("🔮 Step 6 – Inference & What-if Analysis")

# Convert scaled X_test back to DataFrame (for selection)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# Reset index of y_test for easy indexing
y_test_series = y_test.reset_index(drop=True)

# Numeric input to pick which test sample index to inspect
sample_index = st.number_input(
    "Select test sample index:",
    min_value=0,
    max_value=len(X_test_df) - 1,
    value=0,
    step=1
)

# When user clicks this button, we predict for the selected sample
if st.button("🔍 Predict for Selected Sample"):
    # Extract 1-row sample from X_test and convert to 2D array
    x_sample = X_test_df.iloc[[sample_index]].values
    # True target for this sample
    true_value = y_test_series.iloc[sample_index]
    # Model prediction for this sample
    pred_value = model.predict(x_sample)[0]

    # Show results differently for classification vs regression
    if task_type == "classification":
        st.write(f"**True label:** {true_value}")
        st.write(f"**Predicted label:** {pred_value}")
    else:
        st.write(f"**True target (e.g., RUL):** {true_value:.3f}")
        st.write(f"**Predicted target:** {pred_value:.3f}")

    # Also show feature values of this sample
    st.write("**Feature values for this sample:**")
    st.write(pd.DataFrame(x_sample, columns=feature_names))

# End of app
