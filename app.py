import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 Hotel Booking Analytics Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/hotel_bookings.csv")
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df['total_guests'] = df['adults'] + df['children'].fillna(0) + df['babies']
    df['total_revenue'] = df['adr'] * df['total_guests']
    df['length_of_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    return df

df = load_data()

# Layout
tab1, tab2, tab3 = st.tabs(["📈 Required Analytics", "✨ Advanced Insights", "📌 Raw Data"])

# ------------------------
# Tab 1: Required Analytics
# ------------------------
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue Trends Over Time")
        monthly_rev = df[df['is_canceled'] == 0].groupby(df['reservation_status_date'].dt.to_period("M"))['total_revenue'].sum()
        fig, ax = plt.subplots(figsize=(8, 3))
        monthly_rev.plot(kind='line', ax=ax)
        ax.set_ylabel("Revenue (€)")
        ax.set_xlabel("Date")
        st.pyplot(fig)

    with col2:
        st.subheader("Cancellation Rate (%)")
        cancel_rate = df['is_canceled'].value_counts(normalize=True)[1] * 100
        st.metric("Cancellation Rate", f"{cancel_rate:.2f} %")

        cancel_by_hotel = df.groupby('hotel')['is_canceled'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 3))
        cancel_by_hotel.plot(kind='bar', ax=ax, color='tomato')
        ax.set_ylabel("Cancellation Rate (%)")
        st.pyplot(fig)

    st.subheader("Geographical Distribution (Top 10 Countries)")
    country_dist = df['country'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.barplot(x=country_dist.values, y=country_dist.index, ax=ax, palette="crest")
    ax.set_xlabel("Number of Bookings")
    st.pyplot(fig)

    st.subheader("Lead Time Distribution")
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.histplot(df['lead_time'], kde=True, bins=40, color="purple", ax=ax)
    st.pyplot(fig)

# ------------------------
# Tab 2: Advanced Insights
# ------------------------
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Revenue by Market Segment")
        rev_by_segment = df[df['is_canceled'] == 0].groupby('market_segment')['total_revenue'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3))
        rev_by_segment.plot(kind='bar', ax=ax, color='seagreen')
        ax.set_ylabel("Revenue (€)")
        st.pyplot(fig)

    with col2:
        st.subheader("Booking Trends by Hotel Type")
        booking_trend = df.groupby([df['reservation_status_date'].dt.to_period("M"), 'hotel']).size().unstack()
        fig, ax = plt.subplots(figsize=(7, 3))
        booking_trend.plot(ax=ax)
        ax.set_ylabel("Number of Bookings")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Cancellation Rate by Country (Top 10)")
        cancel_by_country = df.groupby('country')['is_canceled'].mean().sort_values(ascending=False).head(10) * 100
        fig, ax = plt.subplots(figsize=(7, 3))
        cancel_by_country.plot(kind='bar', ax=ax, color='red')
        ax.set_ylabel("Cancellation Rate (%)")
        st.pyplot(fig)

    with col4:
        st.subheader("Avg. Lead Time by Market Segment")
        lead_time_seg = df.groupby('market_segment')['lead_time'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7, 3))
        lead_time_seg.plot(kind='barh', ax=ax, color='teal')
        st.pyplot(fig)

    st.subheader("Family Booking Trend (With Kids/Babies)")
    df['is_family'] = (df['children'] + df['babies']) > 0
    family_trend = df[df['is_family']].groupby(df['reservation_status_date'].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(10, 3))
    family_trend.plot(ax=ax)
    ax.set_ylabel("Bookings with Kids/Babies")
    st.pyplot(fig)

    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Length of Stay Distribution")
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.histplot(df['length_of_stay'], kde=True, bins=30, ax=ax, color='orange')
        st.pyplot(fig)

    with col6:
        st.subheader("Revenue by Country (Top 10)")
        rev_by_country = df[df['is_canceled'] == 0].groupby('country')['total_revenue'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(7, 3))
        rev_by_country.plot(kind='barh', ax=ax, color='skyblue')
        st.pyplot(fig)

# ------------------------
# Tab 3: Raw Data
# ------------------------
with tab3:
    st.subheader("Raw Booking Data")
    st.dataframe(df.sample(100))

