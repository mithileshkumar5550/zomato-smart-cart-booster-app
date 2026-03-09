import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model_training import train_model
from inference import recommend
import random
import pyttsx3

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Zomato Smart Cart Booster",
    page_icon="🍽️",
    layout="wide"
)


# Session State Initialization
if "cart_count" not in st.session_state:
    st.session_state.cart_count = 0

if "voice_summary" not in st.session_state:
    st.session_state.voice_summary = ""

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None


# Banner & Title
st.image(
    "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
    width="stretch"
)

st.markdown(
    "<h1 style='text-align: center; color:#FF4B4B;'>🍽️ Zomato Smart Cart Booster</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align: center;'>AI-Driven Personalized Add-On Recommendation Engine</h4>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; font-size:16px;'>🚀 Increasing Average Order Value through Context-Aware Intelligence</p>",
    unsafe_allow_html=True
)

st.markdown("---")



# Load Model
@st.cache_resource
def load_model():
    return train_model()

model = load_model()


# Load Data
items = pd.read_csv("items.csv")


# Sidebar
st.sidebar.markdown(f"### 🛒 Cart Items: {st.session_state.cart_count}")
st.sidebar.header("📊 Model Performance")
st.sidebar.write("AUC: ~0.50")
st.sidebar.write("Precision@10: ~0.50")

# ==============================
# User Inputs
# ==============================

budget_segment = st.selectbox("Select Budget Type", ["budget", "premium"])
veg_preference = st.selectbox("Veg Preference", ["Veg", "Non-Veg"])
hour = st.slider("Select Hour of Day", 8, 23, 20)
top_k = st.slider("Number of Recommendations", 5, 15, 10)

veg_preference = 1 if veg_preference == "Veg" else 0

# ==============================
# Category Images
# ==============================

category_images = {
    "main": "https://images.unsplash.com/photo-1600891964599-f61ba0e24092",
    "side": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c",
    "drink": "https://images.unsplash.com/photo-1582450871972-ab5ca641643d",
    "dessert": "https://images.unsplash.com/photo-1551024601-bec78aea704b"
}

# ==============================
# Generate Recommendations
# ==============================

if st.button("🚀 Generate Smart Recommendations"):

    user_features = {
        "budget_segment": budget_segment,
        "veg_preference": veg_preference
    }

    with st.spinner("Generating AI-powered recommendations... 🤖"):
        st.session_state.recommendations = recommend(
            model, user_features, items, hour
        )

# ==============================
# Show Recommendations
# ==============================

if st.session_state.recommendations is not None:

    st.markdown("## 🎯 Top Personalized Recommendations")

    total_price = 0
    st.session_state.voice_summary = ""

    for item_id, prob in st.session_state.recommendations[:top_k]:

        item_row = items[items["item_id"] == item_id].iloc[0]
        category = item_row["category"].lower()
        price = item_row["price"]
        total_price += price

        # Build voice summary
        st.session_state.voice_summary += (
            f"Item {item_id}, category {category}, priced at {price} rupees. "
        )

        col1, col2 = st.columns([1, 3])

        with col1:
            image_url = category_images.get(category)
            if image_url:
                st.image(image_url, width="stretch")

        with col2:
            st.markdown(f"""
            ### 🍽️ Item ID: {item_id}
            **Category:** {category.title()}  
            **Price:** ₹{price}
            """)

            st.metric("Confidence", f"{round(float(prob)*100,1)}%")
            st.progress(float(prob))

            if st.button(f"🛒 Add Item {item_id}", key=f"cart_{item_id}"):
                st.session_state.cart_count += 1

        st.markdown("---")

    # ==============================
    # REAL Voice Assistant (pyttsx3)
    # ==============================

    st.markdown("## 🎙 Real Voice Assistant")

    if st.button("🔊 Speak Recommendations", key="real_voice"):

        if st.session_state.voice_summary.strip() == "":
            st.warning("Please generate recommendations first.")
        else:
            engine = pyttsx3.init()
            engine.say(st.session_state.voice_summary)
            engine.runAndWait()

    # ==============================
    # Business Impact
    # ==============================

    avg_price = total_price / top_k
    estimated_uplift = round(avg_price * 0.15, 2)

    st.markdown("## 💰 Business Impact Simulation")

    colA, colB, colC = st.columns(3)
    colA.metric("Avg Recommended Price", f"₹{round(avg_price,2)}")
    colB.metric("Estimated AOV Lift", f"+₹{estimated_uplift}")
    colC.metric("Projected Revenue Boost", "+15%")

    st.markdown("## 📈 Revenue Impact Visualization")

    fig = plt.figure()
    plt.bar(
        ["Before AI", "After AI"],
        [avg_price, avg_price + estimated_uplift]
    )
    plt.ylabel("Average Order Value (₹)")
    st.pyplot(fig)