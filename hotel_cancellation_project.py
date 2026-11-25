# -*- coding: utf-8 -*-
"""
Predicting Hotel Reservation Cancellations – Streamlit app
CIS 412 – Business Machine Learning · Team Project Phase 2 · Team 99% Confidence

Run locally:   streamlit run hotel_cancellation_project.py
Needs Hotel_Reservations.csv in the same folder.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score)

st.set_page_config(page_title="Hotel Cancellation Prediction", layout="wide")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
random_state = 42
BLUE, ORANGE = "#1F4E79", "#E07B39"

# Use 2 workers at most: Streamlit Community Cloud has ~1 GB of RAM, and every
# extra worker holds its own copy of the data and the forests.
N_JOBS = 2


def show(fig):
    """Draw a matplotlib figure on the page, then free its memory."""
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# =============================================================================
# Heavy work – cached so it only runs once per server, not on every page visit
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv("Hotel_Reservations.csv")


@st.cache_data(show_spinner=False)
def prepare_data(hotel_df):
    clean_df = hotel_df.copy()
    rows_before = len(clean_df)

    clean_df = clean_df.drop(columns=["Booking_ID"])

    arrival_dates = pd.to_datetime(
        dict(year=clean_df["arrival_year"], month=clean_df["arrival_month"], day=clean_df["arrival_date"]),
        errors="coerce")
    clean_df = clean_df[arrival_dates.notnull()]
    clean_df = clean_df[(clean_df["no_of_weekend_nights"] + clean_df["no_of_week_nights"]) > 0]
    clean_df = clean_df[clean_df["avg_price_per_room"] <= 500]
    rows_after = len(clean_df)

    clean_df["total_nights"] = clean_df["no_of_weekend_nights"] + clean_df["no_of_week_nights"]
    clean_df["total_guests"] = clean_df["no_of_adults"] + clean_df["no_of_children"]
    clean_df = clean_df.drop(columns=["arrival_year", "arrival_date"])
    clean_df["booking_status"] = clean_df["booking_status"].map({"Canceled": 1, "Not_Canceled": 0})

    text_columns = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
    model_df = pd.get_dummies(clean_df, columns=text_columns, drop_first=True, dtype=int)
    return clean_df, model_df, rows_before, rows_after


@st.cache_resource(show_spinner=False)
def train_everything(model_df):
    X = model_df.drop(columns=["booking_status"])
    y = model_df["booking_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=random_state, n_jobs=N_JOBS),
        "Linear SVM": LinearSVC(random_state=random_state, max_iter=5000),
    }
    for model in models.values():
        model.fit(X_train_scaled, y_train)

    tree_search = GridSearchCV(DecisionTreeClassifier(random_state=random_state),
                               {"max_depth": [5, 10, 15, 20], "min_samples_leaf": [1, 5, 10]},
                               cv=5, scoring="f1", n_jobs=N_JOBS)
    tree_search.fit(X_train_scaled, y_train)

    forest_search = GridSearchCV(RandomForestClassifier(random_state=random_state),
                                 {"n_estimators": [100, 200], "max_depth": [None, 20]},
                                 cv=5, scoring="f1", n_jobs=N_JOBS)
    forest_search.fit(X_train_scaled, y_train)

    final_model = forest_search.best_estimator_
    cv_scores = cross_val_score(final_model, X_train_scaled, y_train, cv=5,
                                scoring="accuracy", n_jobs=N_JOBS)

    return dict(X=X, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled,
                y_train=y_train, y_test=y_test, scaler=scaler, models=models,
                tree_search=tree_search, forest_search=forest_search,
                final_model=final_model, cv_scores=cv_scores)


@st.cache_resource(show_spinner=False)
def run_clustering(clean_df):
    cluster_features = ["lead_time", "avg_price_per_room", "total_nights", "total_guests", "no_of_special_requests"]
    cluster_scaled = StandardScaler().fit_transform(clean_df[cluster_features])
    wcss = []
    for k in range(1, 9):
        wcss.append(KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(cluster_scaled).inertia_)
    labels = KMeans(n_clusters=4, n_init=10, random_state=random_state).fit_predict(cluster_scaled)
    return cluster_features, wcss, labels


def model_scores(model, model_name, X_data):
    if model_name == "Linear SVM":
        return model.decision_function(X_data)
    return model.predict_proba(X_data)[:, 1]


# =============================================================================
# Page
# =============================================================================

st.title("Predicting Hotel Reservation Cancellations")
st.markdown("""
**CIS 412 – Business Machine Learning · Team Project Phase 2**
**Team:** 99% Confidence — Vedika Thareja, Rhys Roberts, Luizeth Denogean, Anthony Caballero, Alexander Salazar, Larissa Backen

**Project goal:** Build a classification model that predicts whether a hotel reservation will be **canceled** or
**not canceled**, find out which booking details matter most, and show how the hotel can use these predictions to
protect revenue. We follow **CRISP-DM**: Business Understanding → Data Understanding → Data Preparation → Modeling →
Evaluation → Deployment.
""")

try:
    hotel_df = load_data()
except FileNotFoundError:
    st.error("Could not find `Hotel_Reservations.csv`. Commit it to the repo in the same folder as this file.")
    st.stop()

with st.spinner("Cleaning data and training models (first load only, this can take a few minutes)..."):
    clean_df, model_df, rows_before, rows_after = prepare_data(hotel_df)
    results = train_everything(model_df)
    cluster_features, wcss_values, cluster_labels = run_clustering(clean_df)

X = results["X"]
X_train_scaled, X_test_scaled = results["X_train_scaled"], results["X_test_scaled"]
y_train, y_test = results["y_train"], results["y_test"]
models, final_model = results["models"], results["final_model"]

tabs = st.tabs(["1. Business", "2. Data", "3. Preparation", "4. Modeling", "5. Evaluation",
                "6. Try the model", "7. Deployment & results", "Appendix: Segments"])

# -----------------------------------------------------------------------------
with tabs[0]:
    st.header("1. Business Understanding")
    st.markdown("""
**The problem:** About one out of every three reservations in our data gets canceled. When a booking is canceled close
to the arrival date, the room often stays empty, and the hotel loses that revenue. Cancellations also make it hard to
plan staffing, food, and how many rooms to overbook.

**How machine learning helps:** If we can predict *which* bookings are likely to cancel, the hotel can act before it
happens. For example, it can:
- send reminders or ask for a deposit on high-risk bookings,
- offer a small incentive to keep the booking,
- overbook a safe number of rooms based on expected cancellations.

**Machine learning question:** This is a **binary classification** problem. Target = `booking_status`
(1 = Canceled, 0 = Not Canceled).

**Expected impact:** Fewer empty rooms, better occupancy forecasts, smarter overbooking and pricing, and better
planning for hotel managers, revenue management, reservations, and marketing teams.
""")

# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("2. Data Understanding")
    hotel_df = hotel_df.copy()
    hotel_df["canceled"] = (hotel_df["booking_status"] == "Canceled").astype(int)

    st.subheader("2.1 Structure and data types")
    c1, c2 = st.columns(2)
    c1.metric("Bookings", f"{hotel_df.shape[0]:,}")
    c2.metric("Columns", hotel_df.shape[1] - 1)
    st.dataframe(hotel_df.drop(columns="canceled").head())
    st.dataframe(pd.DataFrame({
        "Data type": hotel_df.drop(columns="canceled").dtypes.astype(str),
        "Non-null count": hotel_df.drop(columns="canceled").notnull().sum(),
    }))
    st.markdown("""
The dataset has **36,275 bookings** and **19 columns**: 1 ID, 17 features, and 1 target. 14 columns are numbers
(like `lead_time` and `avg_price_per_room`) and 5 are text (meal plan, room type, market segment, booking status, and the ID).
""")

    st.subheader("2.2 Summary statistics")
    st.dataframe(hotel_df.drop(columns="canceled").describe().round(2).T)
    st.markdown("""
The average booking is made about **85 days** ahead and costs about **$103 per night**. Lead time goes up to 443 days
and price goes up to $540, so there are some extreme values. Some bookings also show **0 nights** or a **$0 price**,
which we look at in Data Preparation.
""")

    st.subheader("2.3 Data quality checks")
    arrival_dates = pd.to_datetime(
        dict(year=hotel_df["arrival_year"], month=hotel_df["arrival_month"], day=hotel_df["arrival_date"]),
        errors="coerce")
    total_nights = hotel_df["no_of_weekend_nights"] + hotel_df["no_of_week_nights"]
    raw = hotel_df.drop(columns="canceled")
    quality = pd.DataFrame({"Count": {
        "Missing values in the whole table": int(raw.isnull().sum().sum()),
        "Exact duplicate rows": int(raw.duplicated().sum()),
        "Rows identical except for Booking ID": int(raw.drop(columns=["Booking_ID"]).duplicated().sum()),
        "Impossible arrival dates": int(arrival_dates.isnull().sum()),
        "Bookings with 0 nights": int((total_nights == 0).sum()),
        "Bookings with a $0 room price": int((hotel_df["avg_price_per_room"] == 0).sum()),
        "Bookings with price over $500": int((hotel_df["avg_price_per_room"] > 500).sum()),
    }})
    st.table(quality)
    st.markdown("""
There are **no missing values** and **no exact duplicates** (every row has its own Booking ID). We did find a few
problems we fix later: arrival dates of **February 29, 2018** (2018 was not a leap year), bookings with **0 total
nights**, and one booking priced at **$540**, far above the others.

**Near-duplicates:** About 10,000 rows match another row in every column except the Booking ID. Because each has its
own Booking ID, we treat them as **separate real bookings** (e.g., a group booking several identical rooms) and keep
them. This could make test scores slightly optimistic, so we list it as a limitation.

**$0 prices:** Most $0 rooms are in the **Complementary** (free) market segment, so they are real and we keep them.
""")

    st.subheader("2.4 Target variable – how many bookings were canceled?")
    status_counts = hotel_df["booking_status"].value_counts()
    status_percent = hotel_df["booking_status"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Not Canceled", "Canceled"], status_counts.values, color=[BLUE, ORANGE])
    for bar, pct in zip(bars, status_percent.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300, f"{pct:.1f}%", ha="center", fontsize=12)
    ax.set_title("How many bookings were canceled?")
    ax.set_ylabel("Number of bookings")
    show(fig)
    st.markdown("""
About **33% of bookings are canceled**, a big revenue risk. The classes are somewhat imbalanced (67/33), so we look at
precision, recall, F1, and AUC, not just accuracy.
""")

    st.subheader("2.5 Does booking far in advance lead to more cancellations?")
    lead_time_groups = pd.cut(hotel_df["lead_time"], bins=[-1, 30, 90, 180, 450],
                              labels=["0-30 days", "31-90 days", "91-180 days", "180+ days"])
    cancel_rate_by_lead = hotel_df.groupby(lead_time_groups, observed=True)["canceled"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(cancel_rate_by_lead.index.astype(str), cancel_rate_by_lead.values, color=BLUE)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.0f}%", ha="center")
    ax.set_title("Cancellation rate by lead time")
    ax.set_xlabel("Days booked in advance")
    ax.set_ylabel("Cancellation rate (%)")
    ax.set_ylim(0, 85)
    show(fig)
    st.markdown("""
Cancellations rise sharply with lead time. Bookings made **within 30 days** cancel about **14%** of the time, while
bookings made **more than 180 days ahead** cancel about **74%** of the time. Lead time is likely a key predictor.
""")

    st.subheader("2.6 Do special requests and market segment matter?")
    cancel_rate_by_requests = hotel_df.groupby("no_of_special_requests")["canceled"].mean() * 100
    cancel_rate_by_segment = hotel_df.groupby("market_segment_type")["canceled"].mean().sort_values(ascending=False) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(cancel_rate_by_requests.index.astype(str), cancel_rate_by_requests.values, color=BLUE)
    axes[0].set_title("Cancellation rate by number of special requests")
    axes[0].set_xlabel("Number of special requests")
    axes[0].set_ylabel("Cancellation rate (%)")
    axes[1].bar(cancel_rate_by_segment.index, cancel_rate_by_segment.values, color=ORANGE)
    axes[1].set_title("Cancellation rate by market segment")
    axes[1].set_xlabel("Market segment")
    axes[1].set_ylabel("Cancellation rate (%)")
    fig.tight_layout()
    show(fig)
    st.markdown("""
Guests with **no special requests** cancel about **43%** of the time, while guests with **3 or more requests almost
never cancel**. **Online** bookings cancel most often (about **37%**), while **Corporate** bookings cancel only about **11%**.
""")

    st.subheader("2.7 Room price and cancellations")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=hotel_df, x="booking_status", y="avg_price_per_room",
                hue="booking_status", palette=[BLUE, ORANGE], legend=False, ax=ax)
    ax.set_title("Room price by booking status")
    ax.set_xlabel("Booking status")
    ax.set_ylabel("Average price per room ($)")
    ax.set_ylim(0, 300)
    show(fig)
    st.markdown("Canceled bookings have a slightly **higher typical price** (median about $108 vs. $95). "
                "Price matters, but less than lead time.")

    st.subheader("2.8 Relationships between numeric variables")
    correlation_table = hotel_df.select_dtypes(include="number").corr()
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(correlation_table, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation between numeric variables")
    show(fig)
    st.markdown("""
`lead_time` has the strongest positive link with cancellation, and `no_of_special_requests` has the strongest
negative link. No two features are so similar that we need to remove one.
""")

# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("3. Data Preparation")
    st.markdown("""
1. Drop the `Booking_ID` column (just a label, not useful for prediction)
2. Remove impossible arrival dates (Feb 29, 2018)
3. Remove bookings with 0 nights (not a real stay)
4. Remove the extreme price outlier (over $500)
5. Feature engineering: create `total_nights` and `total_guests`
6. Drop `arrival_year` and `arrival_date` (day of month); keep `arrival_month` for seasonality
7. Turn the target into 1 = Canceled, 0 = Not Canceled
8. One-hot encode the text columns (meal plan, room type, market segment), using `drop_first=True`
9. Split into 80% training and 20% testing data (`stratify=y` keeps the 33% cancel rate in both)
10. Scale the numbers with `StandardScaler`, **fit on the training data only** so no test information leaks in

There were **no missing values**, so no imputation was needed.
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before cleaning", f"{rows_before:,}")
    c2.metric("Rows after cleaning", f"{rows_after:,}")
    c3.metric("Rows removed", rows_before - rows_after)

    c1, c2, c3 = st.columns(3)
    c1.metric("Training rows", f"{X_train_scaled.shape[0]:,}")
    c2.metric("Testing rows", f"{X_test_scaled.shape[0]:,}")
    c3.metric("Features", X_train_scaled.shape[1])
    st.write(f"Cancellation rate — training set: **{y_train.mean() * 100:.1f}%**, "
             f"testing set: **{y_test.mean() * 100:.1f}%**")
    st.write("First rows after encoding:")
    st.dataframe(model_df.head())

# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("4. Modeling")
    st.markdown("""
**Type of problem:** Supervised **binary classification** (Canceled = 1 or Not Canceled = 0).

| Model | Simple explanation |
|---|---|
| **Decision Tree** | Asks a series of yes/no questions (e.g., "Is lead time over 150 days?") to reach a prediction. Easy to explain. |
| **Logistic Regression** | Gives each feature a weight and turns the total into a probability of cancellation. Simple baseline. |
| **Random Forest** | An **ensemble** of many decision trees that each vote. Usually more accurate and less likely to overfit than one tree. |
| **Linear SVM** | Finds the best dividing line between canceled and kept bookings. We use the linear version because regular SVM is very slow on 29,000 rows. |

**Metrics:** **Accuracy** (% of all predictions correct), **Precision** (of bookings we flag, % that really cancel),
**Recall** (of bookings that really cancel, % we catch), **F1** (balance of precision and recall).
""")

    st.subheader("4.1 Performance on the training set")
    training_results = []
    for model_name, model in models.items():
        p = model.predict(X_train_scaled)
        training_results.append({"Model": model_name,
                                 "Accuracy": accuracy_score(y_train, p),
                                 "Precision": precision_score(y_train, p),
                                 "Recall": recall_score(y_train, p),
                                 "F1": f1_score(y_train, p)})
    training_table = pd.DataFrame(training_results).set_index("Model").round(3)
    st.dataframe(training_table)
    st.markdown("""
- The **Decision Tree** and **Random Forest** score almost perfectly on training data — a warning sign of **overfitting**.
- **Logistic Regression** and **Linear SVM** score around 80%. They cannot memorize as easily, so their training
  scores are a more honest guess of how they will do on new data.
""")

# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("5. Evaluation")

    st.subheader("5.1 Performance on the test set")
    test_results = []
    for model_name, model in models.items():
        p = model.predict(X_test_scaled)
        test_results.append({"Model": model_name,
                             "Accuracy": accuracy_score(y_test, p),
                             "Precision": precision_score(y_test, p),
                             "Recall": recall_score(y_test, p),
                             "F1": f1_score(y_test, p),
                             "AUC": roc_auc_score(y_test, model_scores(model, model_name, X_test_scaled))})
    test_table = pd.DataFrame(test_results).set_index("Model").round(3)
    st.dataframe(test_table)
    st.markdown("**AUC** measures how well the model ranks risky bookings above safe ones across all thresholds "
                "(0.5 = random guessing, 1.0 = perfect).")

    st.subheader("5.2 Training vs. test – checking for overfitting")
    accuracy_compare = pd.DataFrame({"Training Accuracy": training_table["Accuracy"],
                                     "Test Accuracy": test_table["Accuracy"]})
    accuracy_compare["Gap"] = (accuracy_compare["Training Accuracy"] - accuracy_compare["Test Accuracy"]).round(3)
    st.dataframe(accuracy_compare)
    fig, ax = plt.subplots(figsize=(8, 4))
    accuracy_compare[["Training Accuracy", "Test Accuracy"]].plot(kind="bar", color=["#9DB4CC", BLUE], ax=ax)
    ax.set_title("Training vs. test accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.5, 1.12)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(loc="upper center", ncol=2)
    show(fig)
    st.markdown("The single **Decision Tree** has the biggest gap. The **Random Forest** also drops, but still has "
                "the **highest test score**, because averaging many trees cancels out much of the memorizing.")

    st.subheader("5.3 ROC curves")
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, model in models.items():
        s = model_scores(model, model_name, X_test_scaled)
        fpr, tpr, _ = roc_curve(y_test, s)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, s):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    show(fig)
    st.markdown("""
The x-axis is the **False Positive Rate** (FP / (FP + TN)) and the y-axis the **True Positive Rate** (TP / (TP + FN)).
The **Random Forest** curve is highest (AUC ≈ 0.95). Logistic Regression and Linear SVM sit almost on top of each other
(AUC ≈ 0.87). The single Decision Tree is lowest because it gives very "yes or no" probabilities.
""")

    st.subheader("5.4 Improving the models – grid search with 5-fold cross-validation")
    tree_search, forest_search = results["tree_search"], results["forest_search"]
    st.markdown("""
`GridSearchCV` tries different settings and picks the best using **5-fold cross-validation** scored on **F1**.
Decision Tree: `max_depth` × `min_samples_leaf`. Random Forest: `n_estimators` × `max_depth`.
""")
    c1, c2 = st.columns(2)
    c1.write(f"**Best Decision Tree settings:** `{tree_search.best_params_}`  \nCV F1: **{tree_search.best_score_:.3f}**")
    c2.write(f"**Best Random Forest settings:** `{forest_search.best_params_}`  \nCV F1: **{forest_search.best_score_:.3f}**")

    tuned_results = []
    for model_name, model in {"Decision Tree (tuned)": tree_search.best_estimator_,
                              "Random Forest (tuned)": forest_search.best_estimator_}.items():
        p = model.predict(X_test_scaled)
        tuned_results.append({"Model": model_name,
                              "Train Accuracy": accuracy_score(y_train, model.predict(X_train_scaled)),
                              "Test Accuracy": accuracy_score(y_test, p),
                              "Precision": precision_score(y_test, p),
                              "Recall": recall_score(y_test, p),
                              "F1": f1_score(y_test, p),
                              "AUC": roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])})
    st.dataframe(pd.DataFrame(tuned_results).set_index("Model").round(3))
    st.markdown("""
- **Decision Tree:** limiting depth cut the training/test gap from about 12.6 points to about 5, and test accuracy
  and AUC both went **up**. Less overfitting = better on new bookings.
- **Random Forest:** limiting depth cut the gap from about 9 points to about 5.5, with test accuracy about the same
  (≈0.90) and AUC slightly higher.
- The tuned Random Forest is still the best on every test metric, so it is our **final model**.
""")

    st.subheader("5.5 Is the result stable? – 5-fold cross-validation of the final model")
    cv_scores = results["cv_scores"]
    st.write(f"Accuracy for each fold: {', '.join(f'{s:.3f}' for s in cv_scores)}")
    c1, c2 = st.columns(2)
    c1.metric("Average CV accuracy", f"{cv_scores.mean():.3f}")
    c2.metric("Standard deviation", f"{cv_scores.std():.3f}")
    st.markdown("The five scores are close together, so one lucky split is not driving the result.")

    st.subheader("5.6 Confusion matrix for the final model")
    final_predictions = final_model.predict(X_test_scaled)
    final_probabilities = final_model.predict_proba(X_test_scaled)[:, 1]
    final_matrix = confusion_matrix(y_test, final_predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(final_matrix, display_labels=["Not Canceled", "Canceled"]).plot(
        cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Confusion matrix – tuned Random Forest (test set)")
    ax.grid(False)
    show(fig)
    st.markdown("""
- **TP:** predicted cancel, really canceled · **FP:** predicted cancel, guest showed up
- **FN:** predicted keep, guest canceled · **TN:** predicted keep, guest showed up

FPs are the costly mistake if the hotel overbooks (a guest arrives and there is no room). FNs are missed revenue.
""")

    st.subheader("5.7 Changing the threshold")
    threshold_results = []
    for threshold in [0.3, 0.5, 0.7]:
        flagged = (final_probabilities >= threshold).astype(int)
        tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, flagged).ravel()
        threshold_results.append({"Threshold": threshold,
                                  "Bookings flagged": int(flagged.sum()),
                                  "True Positives": int(tp_t),
                                  "False Positives": int(fp_t),
                                  "Precision": round(precision_score(y_test, flagged), 3),
                                  "Recall": round(recall_score(y_test, flagged), 3)})
    st.dataframe(pd.DataFrame(threshold_results), hide_index=True)
    st.markdown("**Lowering** the threshold catches more cancellations (higher recall) but flags more guests who "
                "would have shown up (lower precision). A cheap action like a reminder email can use a **low** "
                "threshold; a risky action like overbooking should use a **high** one.")

    st.subheader("5.8 Cumulative response and lift")
    ranking_df = pd.DataFrame({"actual": y_test.values, "probability": final_probabilities})
    ranking_df = ranking_df.sort_values("probability", ascending=False).reset_index(drop=True)
    total_bookings = len(ranking_df)
    total_cancellations = ranking_df["actual"].sum()
    overall_cancel_rate = total_cancellations / total_bookings
    ranking_df["percent_targeted"] = (ranking_df.index + 1) / total_bookings * 100
    ranking_df["percent_cancellations_caught"] = ranking_df["actual"].cumsum() / total_cancellations * 100
    ranking_df["lift"] = (ranking_df["actual"].cumsum() / (ranking_df.index + 1)) / overall_cancel_rate

    lift_rows = []
    for percent in [10, 20, 30, 50]:
        r = int(total_bookings * percent / 100) - 1
        lift_rows.append({"Top % of bookings targeted": percent,
                          "% of all cancellations caught": round(ranking_df.loc[r, "percent_cancellations_caught"], 1),
                          "Lift": round(ranking_df.loc[r, "lift"], 2)})
    st.dataframe(pd.DataFrame(lift_rows), hide_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ranking_df["percent_targeted"], ranking_df["percent_cancellations_caught"], color=BLUE, label="Random Forest")
    axes[0].plot([0, 100], [0, 100], linestyle="--", color="gray", label="Random")
    axes[0].set_title("Cumulative response curve")
    axes[0].set_xlabel("% of bookings targeted (riskiest first)")
    axes[0].set_ylabel("% of all cancellations caught")
    axes[0].legend()
    axes[1].plot(ranking_df["percent_targeted"], ranking_df["lift"], color=ORANGE, label="Random Forest")
    axes[1].axhline(1, linestyle="--", color="gray", label="Random (lift = 1)")
    axes[1].set_title("Lift curve")
    axes[1].set_xlabel("% of bookings targeted (riskiest first)")
    axes[1].set_ylabel("Lift")
    axes[1].set_xlim(1, 100)
    axes[1].legend()
    fig.tight_layout()
    show(fig)
    st.markdown("A lift **greater than 1** means the model beats random selection. Contacting just the top 10% "
                "riskiest bookings reaches cancellations about **3 times** more often than picking at random.")

    st.subheader("5.9 Profit curve – how many rooms should the hotel overbook?")
    c1, c2 = st.columns(2)
    benefit_true_positive = c1.number_input("Benefit per correct overbooking ($)", min_value=0.0,
                                            value=float(round(clean_df["avg_price_per_room"].mean(), 2)), step=5.0)
    cost_false_positive = c2.number_input("Cost per wrong overbooking ($)", min_value=0.0, value=150.0, step=10.0)

    ranking_df["profit"] = (ranking_df["actual"].cumsum() * benefit_true_positive
                            - (1 - ranking_df["actual"]).cumsum() * cost_false_positive)
    best_row = ranking_df["profit"].idxmax()
    best_percent = ranking_df.loc[best_row, "percent_targeted"]
    best_profit = ranking_df.loc[best_row, "profit"]
    best_threshold = ranking_df.loc[best_row, "probability"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Best % of bookings to target", f"{best_percent:.1f}%")
    c2.metric("Probability cutoff", f"{best_threshold:.2f}")
    c3.metric("Estimated profit (test set)", f"${best_profit:,.0f}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ranking_df["percent_targeted"], ranking_df["profit"] / 1000, color=BLUE)
    ax.axhline(0, color="gray", linestyle="--")
    ax.scatter(best_percent, best_profit / 1000, color=ORANGE, zorder=5, s=60,
               label=f"Best: target top {best_percent:.0f}%  (${best_profit / 1000:,.0f}K)")
    ax.set_title("Profit curve – overbooking with the Random Forest")
    ax.set_xlabel("% of bookings targeted (riskiest first)")
    ax.set_ylabel("Profit on test set ($ thousands)")
    ax.legend()
    show(fig)
    st.markdown("""
With the default assumptions (gain about one night's revenue, ~$103, per correct overbooking; pay ~$150 to relocate a
guest who shows up), profit peaks around the **riskiest 27%** of bookings and then falls as we start overbooking guests
who will show up. *These dollar values are illustrations; change them above to see how the best cutoff moves.*
""")

    st.subheader("5.10 Which booking details matter most?")
    importance_table = pd.DataFrame({"Feature": X.columns,
                                     "Importance": final_model.feature_importances_}
                                    ).sort_values("Importance", ascending=False)
    top_ten = importance_table.head(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_ten["Feature"][::-1], top_ten["Importance"][::-1], color=BLUE)
    ax.set_title("Top 10 features – tuned Random Forest")
    ax.set_xlabel("Importance")
    show(fig)
    st.markdown("**Lead time**, **room price**, **number of special requests**, **arrival month**, and the "
                "**Online** segment are the strongest predictors, matching what we saw in the charts.")

# -----------------------------------------------------------------------------
with tabs[5]:
    st.header("6. Try the model on a booking")
    st.markdown("Enter a booking's details to get the tuned Random Forest's **chance of cancellation**, "
                "a **risk level**, and a **suggested action**.")

    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        lead_time = c1.number_input("Lead time (days booked ahead)", 0, 500, 85)
        price = c1.number_input("Average price per room ($)", 0.0, 500.0, 103.0, step=5.0)
        arrival_month = c1.slider("Arrival month", 1, 12, 6)
        adults = c2.number_input("Adults", 0, 4, 2)
        children = c2.number_input("Children", 0, 10, 0)
        weekend_nights = c2.number_input("Weekend nights", 0, 7, 1)
        week_nights = c2.number_input("Week nights", 0, 17, 2)
        special_requests = c3.number_input("Special requests", 0, 5, 0)
        parking = c3.checkbox("Needs car parking")
        repeated = c3.checkbox("Repeat guest")
        prev_cancel = c3.number_input("Previous cancellations", 0, 13, 0)
        prev_kept = c3.number_input("Previous bookings not canceled", 0, 58, 0)
        meal = c1.selectbox("Meal plan", sorted(hotel_df["type_of_meal_plan"].unique()))
        room = c2.selectbox("Room type", sorted(hotel_df["room_type_reserved"].unique()))
        segment = c3.selectbox("Market segment", sorted(hotel_df["market_segment_type"].unique()),
                               index=sorted(hotel_df["market_segment_type"].unique()).index("Online"))
        submitted = st.form_submit_button("Predict")

    if submitted:
        if weekend_nights + week_nights == 0:
            st.warning("A booking needs at least one night.")
        else:
            booking = pd.DataFrame([{
                "no_of_adults": adults, "no_of_children": children,
                "no_of_weekend_nights": weekend_nights, "no_of_week_nights": week_nights,
                "required_car_parking_space": int(parking), "lead_time": lead_time,
                "arrival_month": arrival_month, "repeated_guest": int(repeated),
                "no_of_previous_cancellations": prev_cancel,
                "no_of_previous_bookings_not_canceled": prev_kept,
                "avg_price_per_room": price, "no_of_special_requests": special_requests,
                "total_nights": weekend_nights + week_nights, "total_guests": adults + children,
                f"type_of_meal_plan_{meal}": 1, f"room_type_reserved_{room}": 1,
                f"market_segment_type_{segment}": 1,
            }])
            # same columns, same order as training; missing dummy columns become 0
            booking = booking.reindex(columns=X.columns, fill_value=0)
            probability = final_model.predict_proba(results["scaler"].transform(booking))[0, 1]

            if probability >= 0.7:
                level, action = "High", "Ask for a deposit or confirmation; candidate for overbooking."
            elif probability >= 0.4:
                level, action = "Medium", "Send a reminder email or confirmation call closer to arrival."
            else:
                level, action = "Low", "No action needed."
            c1, c2 = st.columns(2)
            c1.metric("Chance of cancellation", f"{probability:.0%}")
            c2.metric("Risk level", level)
            st.info(f"**Suggested action:** {action}")

# -----------------------------------------------------------------------------
with tabs[6]:
    st.header("7. Deployment")
    st.markdown("""
This Streamlit app is our deployment: the **Try the model** tab lets a hotel employee enter a booking and see the
chance of cancellation, a risk level, and a suggested action, using the same cleaning steps and tuned Random Forest
as the analysis.

**How the hotel would use it:**
- Score every new booking each night and send a list of high-risk bookings to the reservations team.
- Low-cost actions (reminder emails, confirmation calls) for medium/high risk; deposits or overbooking only for the highest-risk bookings.
- Revenue management uses the predicted number of cancellations per night to set overbooking levels.

**Ethical considerations:**
- **Fairness:** The model should not lead to treating guests worse (e.g., higher prices) only because of how they booked. Use it to *help* guests keep bookings, not punish them.
- **Privacy:** The data has no names or personal IDs; any live system should keep guest data secure and follow privacy rules.
- **Transparency:** Staff should know the score is a prediction, not a fact.

| Risk | How we reduce it |
|---|---|
| Overbooking a guest who shows up (False Positive) | Only overbook above a high threshold; keep a relocation plan with nearby hotels |
| Model gets out of date (travel patterns change) | Retrain every few months with new bookings and monitor accuracy |
| Data only covers 2017–2018 at one hotel | Test the model on newer data before relying on it |
| Near-duplicate bookings may make scores look slightly better | Check performance again on new, real bookings after launch |
""")

    st.header("8. Results and Business Recommendations")
    st.markdown("""
**Results**
- The **tuned Random Forest** is our best model: on the test set it has **89.9% accuracy, 0.88 precision, 0.80 recall, 0.84 F1, and 0.958 AUC**.
- The single Decision Tree overfit badly (99% training vs 87% test); tuning `max_depth` reduced the gap.
- 5-fold cross-validation accuracy was **0.893 ± 0.005**, so the result is stable.
- Contacting the riskiest 10% of bookings reaches cancellations about **3 times** more often than random (lift = 3.0), and the riskiest 30% contains about **80%** of all cancellations.

**Recommendations for the hotel**
1. **Watch long lead times.** Bookings made 90+ days ahead are the riskiest. Send reminders or ask for a deposit as the date gets closer.
2. **Encourage special requests.** Guests who add requests rarely cancel. Prompting guests to personalize their stay during online booking may increase commitment.
3. **Tighten online booking policies.** The Online segment has the highest cancellation rate; consider non-refundable discounts for online bookings.
4. **Use the model to set overbooking levels**, using the profit curve to decide how many rooms are safe to overbook.
5. **Retrain the model regularly** and replace our example costs with the hotel's real numbers.
""")

# -----------------------------------------------------------------------------
with tabs[7]:
    st.header("Appendix: Guest Segments with K-Means Clustering")
    st.markdown("An extra, **unsupervised** analysis: instead of predicting cancellations, we group bookings into "
                "similar **segments** and check the cancellation rate in each one.")

    st.subheader("A.1 Choosing k with the elbow method")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 9), wcss_values, marker="o", color=BLUE)
    ax.set_title("Elbow method for choosing k")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("WCSS")
    show(fig)
    st.markdown("""
**WCSS** (Within-Cluster Sum of Squares) measures how tightly packed each cluster is. We use `n_init=10` so k-means
runs 10 times from different starting centroids and keeps the best. The biggest drops happen up to about k = 3–4, so we
choose **k = 4**, which still gives a few easy-to-explain guest types.
""")

    st.subheader("A.2 Describing the clusters")
    segment_df = clean_df.copy()
    segment_df["cluster"] = cluster_labels
    cluster_summary = segment_df.groupby("cluster")[cluster_features].mean().round(1)
    cluster_summary["bookings"] = segment_df["cluster"].value_counts().sort_index()
    cluster_summary["cancel_rate_%"] = (segment_df.groupby("cluster")["booking_status"].mean() * 100).round(1)
    st.dataframe(cluster_summary)
    st.markdown("""
- **Early planners** – book about 7 months ahead with few special requests. Cancel about **69%** of the time, the highest by far.
- **Families / bigger groups** – about 3 guests and the highest room price (~$158). Cancel about **38%** of the time.
- **Short-notice, short stays** – book about 6 weeks ahead for about 2 nights, the largest group. Cancel about **25%** of the time.
- **Longer stays with requests** – about 4 nights and the most special requests. The most reliable, cancelling about **17%** of the time.

This supports our model: long lead time and few special requests = high risk. Marketing can target the
"early planners" with reminders or non-refundable deals.
""")
