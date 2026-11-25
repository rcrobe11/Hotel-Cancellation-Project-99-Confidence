"""
Hotel Cancellation Predictor - CIS 412 Team Project (99% Confidence)

This app uses the same cleaning steps and the same tuned Random Forest
as our Colab notebook. A hotel employee enters a booking and the app shows
the chance that the booking will be canceled.
"""

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="wide")

random_state = 42
text_columns = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]


# ---------------------------------------------------------------
# 1. Load and clean the data (same steps as the notebook)
# ---------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    hotel_df = pd.read_csv("Hotel_Reservations.csv")
    clean_df = hotel_df.drop(columns=["Booking_ID"])

    # remove impossible dates (Feb 29, 2018)
    arrival_dates = pd.to_datetime(
        dict(year=clean_df["arrival_year"], month=clean_df["arrival_month"], day=clean_df["arrival_date"]),
        errors="coerce")
    clean_df = clean_df[arrival_dates.notnull()]

    # remove 0-night bookings and the extreme price outlier
    clean_df = clean_df[(clean_df["no_of_weekend_nights"] + clean_df["no_of_week_nights"]) > 0]
    clean_df = clean_df[clean_df["avg_price_per_room"] <= 500]

    # feature engineering
    clean_df["total_nights"] = clean_df["no_of_weekend_nights"] + clean_df["no_of_week_nights"]
    clean_df["total_guests"] = clean_df["no_of_adults"] + clean_df["no_of_children"]
    clean_df = clean_df.drop(columns=["arrival_year", "arrival_date"])

    # target: 1 = Canceled, 0 = Not Canceled
    clean_df["booking_status"] = clean_df["booking_status"].map({"Canceled": 1, "Not_Canceled": 0})
    return clean_df


# ---------------------------------------------------------------
# 2. Train the final model (tuned Random Forest from the notebook)
# ---------------------------------------------------------------
@st.cache_resource
def train_model():
    clean_df = load_and_clean_data()
    model_df = pd.get_dummies(clean_df, columns=text_columns, drop_first=True, dtype=int)

    X = model_df.drop(columns=["booking_status"])
    y = model_df["booking_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=random_state, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # test-set results so the app matches the notebook
    test_predictions = model.predict(X_test_scaled)
    test_probabilities = model.predict_proba(X_test_scaled)[:, 1]
    results = {
        "Accuracy": accuracy_score(y_test, test_predictions),
        "Precision": precision_score(y_test, test_predictions),
        "Recall": recall_score(y_test, test_predictions),
        "F1": f1_score(y_test, test_predictions),
        "AUC": roc_auc_score(y_test, test_probabilities),
    }
    matrix = confusion_matrix(y_test, test_predictions)
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, scaler, list(X.columns), results, matrix, importance


with st.spinner("Training the model (first load only, about 30 seconds)..."):
    clean_df = load_and_clean_data()
    model, scaler, feature_columns, results, matrix, importance = train_model()


# ---------------------------------------------------------------
# 3. Page layout
# ---------------------------------------------------------------
st.title("🏨 Hotel Reservation Cancellation Predictor")
st.write("Enter a booking's details in the sidebar to see how likely it is to be canceled. "
         "Built by **99% Confidence** for CIS 412 using a tuned Random Forest model.")

st.sidebar.header("Booking details")
lead_time = st.sidebar.slider("Lead time (days between booking and arrival)", 0, 443, 60)
arrival_month = st.sidebar.selectbox("Arrival month", list(range(1, 13)), index=5,
                                     format_func=lambda m: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                                            "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1])
avg_price = st.sidebar.number_input("Average price per room ($)", min_value=0.0, max_value=500.0, value=100.0, step=5.0)
special_requests = st.sidebar.slider("Number of special requests", 0, 5, 0)
market_segment = st.sidebar.selectbox("Market segment", ["Online", "Offline", "Corporate", "Aviation", "Complementary"])
adults = st.sidebar.slider("Adults", 0, 4, 2)
children = st.sidebar.slider("Children", 0, 10, 0)
weekend_nights = st.sidebar.slider("Weekend nights", 0, 7, 1)
week_nights = st.sidebar.slider("Week nights", 0, 17, 2)
meal_plan = st.sidebar.selectbox("Meal plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
room_type = st.sidebar.selectbox("Room type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4",
                                               "Room_Type 5", "Room_Type 6", "Room_Type 7"])
parking = st.sidebar.checkbox("Needs a parking space")
repeated_guest = st.sidebar.checkbox("Repeat guest")
previous_cancellations = st.sidebar.number_input("Previous cancellations", 0, 13, 0)
previous_stays = st.sidebar.number_input("Previous stays (not canceled)", 0, 58, 0)

tab_predict, tab_model, tab_data = st.tabs(["Predict a booking", "Model performance", "Data insights"])

# ---------------------------------------------------------------
# Tab 1: prediction
# ---------------------------------------------------------------
with tab_predict:
    if weekend_nights + week_nights == 0:
        st.warning("A booking needs at least 1 night. Please add weekend or week nights.")
    else:
        new_booking = pd.DataFrame([{
            "no_of_adults": adults,
            "no_of_children": children,
            "no_of_weekend_nights": weekend_nights,
            "no_of_week_nights": week_nights,
            "type_of_meal_plan": meal_plan,
            "required_car_parking_space": int(parking),
            "room_type_reserved": room_type,
            "lead_time": lead_time,
            "arrival_month": arrival_month,
            "market_segment_type": market_segment,
            "repeated_guest": int(repeated_guest),
            "no_of_previous_cancellations": previous_cancellations,
            "no_of_previous_bookings_not_canceled": previous_stays,
            "avg_price_per_room": avg_price,
            "no_of_special_requests": special_requests,
            "total_nights": weekend_nights + week_nights,
            "total_guests": adults + children,
        }])

        # same one-hot encoding as training, then line up the columns
        new_booking = pd.get_dummies(new_booking, columns=text_columns, dtype=int)
        new_booking = new_booking.reindex(columns=feature_columns, fill_value=0)
        cancel_probability = model.predict_proba(scaler.transform(new_booking))[0, 1]

        # risk levels: 0.59 is the best cutoff from our profit curve
        if cancel_probability >= 0.59:
            risk_level, color = "High", "🔴"
            action = ("Ask for a deposit or send a personal confirmation call. "
                      "This booking is a candidate for overbooking.")
        elif cancel_probability >= 0.30:
            risk_level, color = "Medium", "🟠"
            action = "Send a reminder email and offer a small perk (e.g., free breakfast) to keep the booking."
        else:
            risk_level, color = "Low", "🟢"
            action = "No action needed. Treat as a confirmed booking."

        col1, col2 = st.columns(2)
        col1.metric("Chance of cancellation", f"{cancel_probability:.0%}")
        col2.metric("Risk level", f"{color} {risk_level}")
        st.progress(float(cancel_probability))
        st.info(f"**Suggested action:** {action}")
        st.caption("Risk levels: Low < 30%, Medium 30–59%, High ≥ 59% (the most profitable cutoff from our profit curve).")

# ---------------------------------------------------------------
# Tab 2: model performance
# ---------------------------------------------------------------
with tab_model:
    st.subheader("Tuned Random Forest – test set results")
    metric_cols = st.columns(5)
    for column, (name, value) in zip(metric_cols, results.items()):
        column.metric(name, f"{value:.3f}")

    st.write("**Confusion matrix (test set, 7,232 bookings)**")
    matrix_df = pd.DataFrame(matrix,
                             index=["Actual: Not Canceled", "Actual: Canceled"],
                             columns=["Predicted: Not Canceled", "Predicted: Canceled"])
    st.dataframe(matrix_df)

    st.write("**Top 10 most important features**")
    st.bar_chart(importance.head(10).sort_values())

# ---------------------------------------------------------------
# Tab 3: data insights
# ---------------------------------------------------------------
with tab_data:
    st.subheader("What drives cancellations?")
    lead_groups = pd.cut(clean_df["lead_time"], bins=[-1, 30, 90, 180, 450],
                         labels=["0-30 days", "31-90 days", "91-180 days", "180+ days"])
    st.write("**Cancellation rate (%) by lead time**")
    st.bar_chart(clean_df.groupby(lead_groups, observed=True)["booking_status"].mean() * 100)

    st.write("**Cancellation rate (%) by number of special requests**")
    st.bar_chart(clean_df.groupby("no_of_special_requests")["booking_status"].mean() * 100)

    st.write("**Cancellation rate (%) by market segment**")
    st.bar_chart(clean_df.groupby("market_segment_type")["booking_status"].mean().sort_values(ascending=False) * 100)
