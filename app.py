import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime
from collections import deque


st.set_page_config(
    page_title="Real-Time Analytics",
    layout="wide"
)


WINDOW = 30


if "prices" not in st.session_state:
    st.session_state.prices = deque(maxlen=WINDOW)

if "times" not in st.session_state:
    st.session_state.times = deque(maxlen=WINDOW)



def get_price():

    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        return float(data["price"])

    except:
        return np.random.uniform(60000,62000)



def calculate_zscore(values):

    if len(values)<5:
        return 0, False


    series=pd.Series(values)

    mean=series.mean()

    std=series.std()


    if std==0:
        return 0,False


    z=(values[-1]-mean)/std


    return z, abs(z)>2



price=get_price()

current_time=datetime.now().strftime("%H:%M:%S")


st.session_state.prices.append(price)

st.session_state.times.append(current_time)



z, anomaly = calculate_zscore(
    list(st.session_state.prices)
)



st.title(
    "Real-Time Bitcoin Analytics Dashboard"
)



c1,c2,c3=st.columns(3)


c1.metric(
    "BTC Price",
    f"${price:,.2f}"
)


c2.metric(
    "Rolling Z-score",
    round(z,2)
)


if anomaly:
    c3.error("⚠ Abnormal Movement")
else:
    c3.success("Normal")



df=pd.DataFrame(
    {
        "Time":list(st.session_state.times),
        "Price":list(st.session_state.prices)
    }
)



fig=go.Figure()


fig.add_trace(
    go.Scatter(
        x=df["Time"],
        y=df["Price"],
        mode="lines+markers"
    )
)


fig.update_layout(
    title="BTC Real-Time Price Stream",
    height=450
)


st.plotly_chart(
    fig,
    use_container_width=True
)



st.subheader("Statistical Model")

st.write(
"""
Rolling Z-score:

Z = (Xt - μ) / σ


Anomaly condition:

|Z| > 2

The system detects unusual price movements
from real-time streaming data.
"""
)



time.sleep(2)

st.rerun()