import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import seaborn as sns
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import sys
import os

# Function to get resource path for PyInstaller compatibility
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# --- Custom CSS for modern button and input styling ---
st.markdown("""
    <style>
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        cursor: pointer;
    }
    div[data-baseweb="input"] > input[type=number] {
        border-radius: 8px !important;
        border: 1.5px solid #4CAF50 !important;
        padding: 8px !important;
        font-size: 1.1em !important;
        font-weight: 500 !important;
        color: #333 !important;
        width: 80px !important;
        text-align: center !important;
    }
    div[data-baseweb="input"] > input[type=number]:focus {
        border-color: #45a049 !important;
        outline: none !important;
        box-shadow: 0 0 5px #45a049 !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        if os.path.exists(resource_path("lotto_history.csv")):
            history = pd.read_csv(resource_path("lotto_history.csv"))
        else:
            st.warning("⚠️ 'lotto_history.csv' not found. Please upload it.")
            history = pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading 'lotto_history.csv': {e}")
        history = pd.DataFrame()

    # Preprocess historical data
    history_sorted, combo_cols = preprocess(history)
    history_set = set(history_sorted)

    try:
        gen_path = resource_path("generated_combos.csv")
        if os.path.exists(gen_path):
            generated = pd.read_csv(gen_path)
        else:
            generated = pd.DataFrame(columns=["Num1", "Num2", "Num3", "Num4", "Num5"])
    except Exception as e:
        st.error(f"Error loading 'generated_combos.csv': {e}")
        generated = pd.DataFrame(columns=["Num1", "Num2", "Num3", "Num4", "Num5"])

    # Auto-generate only if empty
    if generated.empty and not history.empty and combo_cols:
        st.toast("🔄 Auto-generating 5 most likely combinations...", icon="🔄")

        # Frequency series
        freq_series = history[combo_cols].values.flatten()
        freq_df = pd.Series(freq_series).value_counts().sort_index()

        # LSTM preparation
        X, y = prepare_sequences(freq_df.values, look_back=5)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=50, verbose=0)

        pred_raw = model.predict(X[-1].reshape(1, 5, 1))
        pred_num = int(round(pred_raw[0][0]))
        pred_num = max(1, min(pred_num, 49))  # ensure within bounds

        # Build top frequency combo set
        freq_numbers_sorted = freq_df.sort_values(ascending=False).index.tolist()

        combos = []
        attempts = 0
        while len(combos) < 5 and attempts < 1000:
            combo = {pred_num}
            for num in freq_numbers_sorted:
                if len(combo) >= 5:
                    break
                if num != pred_num:
                    combo.add(num)
            combo = tuple(sorted(combo))
            if combo not in history_set and combo not in combos:
                combos.append(combo)
            else:
                combo = generate_prediction_combo(freq_df, history_set, combo_size=5)
                if combo and combo not in combos:
                    combos.append(combo)
            attempts += 1

        generated = pd.DataFrame(combos, columns=["Num1", "Num2", "Num3", "Num4", "Num5"])
        generated.to_csv(gen_path, index=False)

        st.toast("✅ Auto-generated and saved 5 combinations.", icon="✅")

    return history, generated





# Robust Preprocess Data
def preprocess(df):
    if df is None or df.empty:
        return pd.Series(dtype=object), []
    combo_cols = [col for col in df.columns if
                  any(x in col.lower() for x in ["num", "number", "col"]) and
                  pd.api.types.is_numeric_dtype(df[col])]

    if len(combo_cols) < 5:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 5:
            combo_cols = numeric_cols[:5]
            st.warning(f"Auto-detection failed. Using fallback numeric columns: {combo_cols}")
        else:
            st.error("Failed to detect at least 5 numeric columns for processing.")
            return pd.Series(dtype=object), []

    sorted_combos = df[combo_cols].apply(lambda row: tuple(sorted(row)), axis=1)
    return sorted_combos, combo_cols

# Check Uniqueness
def validate_combos(history_sorted, combos_to_check):
    matches_mask = combos_to_check.isin(set(history_sorted))
    total_matches = matches_mask.sum()
    return total_matches, matches_mask

# Cluster Analysis
def cluster_analysis(df, combo_cols, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    X = df[combo_cols].values
    clusters = kmeans.fit_predict(X)
    return clusters

# Most Frequent Numbers
def most_common_numbers(df, combo_cols):
    all_numbers = df[combo_cols].values.flatten()
    counts = Counter(all_numbers)
    return pd.DataFrame(counts.most_common(), columns=["Number", "Frequency"])

# Generate unique combos from top frequent numbers (1 to 49, no repetition)
def generate_unique_combos(freq_df, history_set, n=5, combo_size=5):
    top_numbers = [num for num in freq_df["Number"].head(49) if 1 <= num <= 49]
    generated = []
    attempts = 0
    max_attempts = 1000
    while len(generated) < n and attempts < max_attempts:
        combo = tuple(sorted(random.sample(top_numbers, combo_size)))
        if (combo not in history_set and combo not in generated and
            all(1 <= x <= 49 for x in combo) and len(set(combo)) == combo_size):
            generated.append(combo)
        attempts += 1
    return generated

# Association Rule Mining
def find_associations(df, combo_cols, min_support=0.01, min_lift=0.7):
    records = df[combo_cols].astype(str).values.tolist()
    te = TransactionEncoder()
    te_ary = te.fit(records).transform(records)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent = apriori(df_encoded, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="lift", min_threshold=min_lift)
    return frequent, rules

# LSTM Preparation and Training
def prepare_sequences(data, look_back=5):
    X, y = [], []
    for i in range(len(data)-look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Helper to generate a valid unique prediction combo (5 unique numbers between 1-49)
def generate_prediction_combo(freq_df, history_set, combo_size=5):
    top_numbers = [num for num in freq_df.index if 1 <= num <= 49]
    generated = None
    attempts = 0
    max_attempts = 500
    while attempts < max_attempts:
        combo = tuple(sorted(random.sample(top_numbers, combo_size)))
        if combo not in history_set:
            generated = combo
            break
        attempts += 1
    return generated

# Function to get most frequent combinations (tuples) and counts
def most_frequent_combinations(history_sorted, top_n=10):
    combo_counts = Counter(history_sorted)
    most_common = combo_counts.most_common(top_n)
    return most_common

# --- Streamlit App ---
st.title("🎯 Lotto Combo Analyzer Dashboard")
st.markdown("Analyze lotto combinations with clustering, prediction, association rules, and frequency analysis.")

combo_size = 5

history, generated = load_data()

if history is None:
    st.warning("Please upload your historical lotto CSV file to get started.")
    uploaded_hist = st.file_uploader("Upload Historical Lotto CSV", type="csv")
    if uploaded_hist:
        history = pd.read_csv(uploaded_hist)
        history.to_csv("lotto_history.csv", index=False)
        st.success("Historical file saved. Please reload the app.")
        st.stop()

history_sorted, combo_cols = preprocess(history)
generated_sorted, _ = preprocess(generated)


option = st.sidebar.radio("Choose an option", [
    "View Data", "Check Uniqueness", "Frequent Numbers",
    "Generate Next Prediction", "Cluster View", "Find Number Patterns",
    "Upload File to Validate", "Input Combo to Validate"
])

if option == "View Data":
    st.subheader("Historical Data")
    if history is not None and not history.empty:
        st.dataframe(history.head())
    else:
        st.warning("⚠️ No historical data loaded or file is missing.")

    st.subheader("Generated Data")
    if generated is not None and not generated.empty:
        st.dataframe(generated.head())
    else:
        st.info("ℹ️ No generated data found yet. Run the generator first.")


elif option == "Check Uniqueness":
    if history_sorted.empty or generated_sorted.empty:
        st.error("Missing columns for validation.")
    else:
        total_matches, matches_mask = validate_combos(history_sorted, generated_sorted)
        st.write(f"Matches found: {total_matches}")
        if total_matches > 0:
            st.dataframe(generated.loc[matches_mask.values])
        else:
            st.success("All combinations are unique.")

elif option == "Frequent Numbers":
    freq_df = most_common_numbers(history, combo_cols)
    st.dataframe(freq_df.head(10))
    fig, ax = plt.subplots()
    sns.barplot(x="Number", y="Frequency", data=freq_df.head(15), palette="viridis", ax=ax, hue="Number", legend=False)
    ax.set_title("Top 15 Most Frequent Numbers")
    st.pyplot(fig)

    st.subheader("Most Frequent Combinations (Top 10)")
    most_common_combos = most_frequent_combinations(history_sorted, top_n=10)
    for combo, count in most_common_combos:
        with st.expander(f"Combo: {', '.join(map(str, combo))} (Count: {count})"):
            st.write(f"This combination appeared {count} times in the past 20 years.")

elif option == "Cluster View":
    clusters = cluster_analysis(history, combo_cols)
    history["Cluster"] = clusters
    fig, ax = plt.subplots()
    sns.countplot(x="Cluster", data=history, palette="tab10", ax=ax, hue="Cluster", legend=False)
    ax.set_title("Combination Clusters")
    st.pyplot(fig)
    st.dataframe(history[[*combo_cols, "Cluster"]].head())

elif option == "Find Number Patterns":
    st.subheader("Association Rules Analysis")
    support_threshold = st.slider("Minimum Support Threshold", 0.0001, 0.05, 0.01, 0.001)
    min_lift = 0.7
    frequent, rules = find_associations(history, combo_cols, min_support=support_threshold, min_lift=min_lift)
    st.write(f"Found {len(frequent)} frequent itemsets with support ≥ {support_threshold}")
    if not rules.empty:
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))
    else:
        st.warning("No rules found with current settings. Try lowering the support threshold.")

elif option == "Generate Next Prediction":
    st.subheader("Predict Next Number Combo (Deep Learning + Frequency Sampling)")
    if not combo_cols:
        st.error("Invalid combo columns.")
    else:
        freq_series = history[combo_cols].values.flatten()
        freq_df = pd.Series(freq_series).value_counts().sort_index()
        X, y = prepare_sequences(freq_df.values, look_back=5)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        if st.button("Generate Prediction"):
            model = build_lstm_model((X.shape[1], 1))
            model.fit(X, y, epochs=50, verbose=0)
            pred_raw = model.predict(X[-1].reshape(1, 5, 1))
            pred_num = int(round(pred_raw[0][0]))
            pred_num = max(1, min(pred_num, 49))

            history_set = set(history_sorted)
            freq_numbers_sorted = freq_df.sort_values(ascending=False).index.tolist()

            combo = {pred_num}
            for num in freq_numbers_sorted:
                if len(combo) >= combo_size:
                    break
                if num != pred_num:
                    combo.add(num)
            combo = tuple(sorted(combo))

            if combo in history_set:
                combo = generate_prediction_combo(freq_df, history_set, combo_size)

            if combo is None:
                st.error("Failed to generate a unique prediction combo.")
            else:
                st.success(f"Next predicted combination: {', '.join(map(str, combo))}")

elif option == "Upload File to Validate":
    uploaded_file = st.file_uploader("Upload your Lotto CSV File", type="csv")
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_sorted, _ = preprocess(uploaded_df)
        history_set = set(history_sorted)
        total_matches, matches_mask = validate_combos(history_set, uploaded_sorted)
        st.success(f"Found {total_matches} matches.")
        if total_matches > 0:
            st.dataframe(uploaded_df.loc[matches_mask.values])
        else:
            st.success("All combinations are unique!")

elif option == "Input Combo to Validate":
    st.subheader("Manual Combo Validation")
    user_input = []
    cols = st.columns(combo_size)
    for i in range(combo_size):
        user_input.append(
            cols[i].number_input(
                f"Number {i+1}",
                min_value=1,
                max_value=49,
                step=1,
                format="%d",
                key=f"manual{i}"
            )
        )

    if st.button("Check Combo"):
        if not all(user_input):
            st.error("Enter all numbers.")
        elif len(set(user_input)) != combo_size:
            st.error("Numbers must be unique and not repeated.")
        else:
            combo = tuple(sorted(user_input))
            if combo in set(history_sorted):
                st.warning("Combo exists in historical data.")
            else:
                st.success("Unique combo!")

st.markdown("---")
st.caption("Created with ❤️ by Randy Costa")
