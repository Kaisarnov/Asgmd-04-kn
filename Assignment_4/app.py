import joblib
import streamlit as st
import pandas as pd

model = joblib.load("model/model.pkl")

st.title("ASG 04 MD - Kaisar - Spaceship Titanic Model Deployment")

st.header("Passenger Features")

age = st.number_input("Age", 0, 100, 25)
roomservice = st.number_input("RoomService", 0.0)
foodcourt = st.number_input("FoodCourt", 0.0)
shoppingmall = st.number_input("ShoppingMall", 0.0)
spa = st.number_input("Spa", 0.0)
vrdeck = st.number_input("VRDeck", 0.0)

homeplanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
cryosleep = st.selectbox("CryoSleep", [True, False])
vip = st.selectbox("VIP", [True, False])

if st.button("Predict"):
    data = pd.DataFrame([{
        "Age": age,
        "RoomService": roomservice,
        "FoodCourt": foodcourt,
        "ShoppingMall": shoppingmall,
        "Spa": spa,
        "VRDeck": vrdeck,
        "HomePlanet": homeplanet,
        "Destination": destination,
        "CryoSleep": cryosleep,
        "VIP": vip
    }])

    prediction = model.predict(data)
    st.write("Prediction:", prediction[0])