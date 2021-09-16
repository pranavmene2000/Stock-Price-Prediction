import streamlit as st
from nsepy import get_history
from datetime import date
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model


def main():
    with st.beta_container():


        html_temp = """
		<p style="color:red;font-size: 20px;font-weight: 700;">{}</p>
		"""

        choose_stock = st.sidebar.selectbox(
            "Choose the Stock!", ["NONE", "Glenmark", "Reliance", "HDFC", "ITC", "NESTLE", "ONGC"])

        bgcolor = st.sidebar.beta_color_picker("Pick a Background color")
        st.markdown("<body style='background-color:yellow;' />",unsafe_allow_html=True)
        
        if(choose_stock == "NONE"):
            st.spinner(text="In Progress...")
            st.title("Real-Time Stocks Predictions")
            st.title("Please... Select stock from sidebar")

        if(choose_stock == "Glenmark"):

            st.title(choose_stock)
            df1 = get_history(symbol='GLENMARK', start=date(
                2015, 1, 1), end=date.today())
            df1['Date'] = df1.index

            new_close_col = df1.filter(['Close'])
            mm_scale = MinMaxScaler(feature_range=(0, 1))
            mm_scale_data = mm_scale.fit_transform(new_close_col)
            new_close_col_val = new_close_col[-60:].values
            new_close_col_val_scale = mm_scale.transform(new_close_col_val)

            X_test = []
            X_test.append(new_close_col_val_scale)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            network = load_model("trained_models/GLENMARK.MODEL")

            new_preds = network.predict(X_test)
            new_preds = mm_scale.inverse_transform(new_preds)

            NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

            st.subheader(
                "Predictions for the next day Close Price : " + str(NextDay_Date))
            formatter = '{:.2f}'.format(new_preds[0][0])
            st.markdown(html_temp.format(formatter),unsafe_allow_html=True)

            st.subheader("Close Price VS Date Interactive chart for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(data=df1['Close'], width=600, height=300,
                          use_container_width=True)

            st.subheader("Line chart of Open and Close for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(df1[['Open', 'Close']])

            st.subheader("Line chart of High and Low for analysis:")
            st.spinner(text='In Progress...')
            st.line_chart(df1[['High', 'Low']])

        if(choose_stock == "Reliance"):

            st.title(choose_stock)
            df1 = pd.read_csv("./CSVS/RELIANCE.csv")
            df1['Date'] = df1.index

            new_close_col = df1.filter(['Close'])
            mm_scale = MinMaxScaler(feature_range=(0, 1))
            mm_scale_data = mm_scale.fit_transform(new_close_col)
            new_close_col_val = new_close_col[-60:].values
            new_close_col_val_scale = mm_scale.transform(new_close_col_val)

            X_test = []
            X_test.append(new_close_col_val_scale)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            network = load_model("Final_Trained_Models/RELIANCE_FINAL.MODEL")

            new_preds = network.predict(X_test)
            new_preds = mm_scale.inverse_transform(new_preds)

            NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

            st.subheader(
                "Predictions for the next day Close Price : 2020-05-30")
            formatter = '{:.2f}'.format(new_preds[0][0])
            st.markdown(html_temp.format(formatter),unsafe_allow_html=True)

            # Plotting Interactive Chart
            st.subheader(
                "Close Price VS Date Interactive chart for analysis : ")
            st.spinner(text='In Progress...')
            st.area_chart(data=df1['Close'], width=600, height=300,
                          use_container_width=True)
            
            st.subheader("Line chart of Open and Close for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(df1[['Open', 'Close']])

            st.subheader("Line chart of High and Low for analysis:")
            st.spinner(text='In Progress...')
            st.line_chart(df1[['High', 'Low']])

        if(choose_stock == "HDFC"):

            st.title(choose_stock)
            df1 = get_history(symbol='hdfc', start=date(
                2015, 1, 1), end=date.today())
            df1['Date'] = df1.index

            new_close_col = df1.filter(['Close'])
            mm_scale = MinMaxScaler(feature_range=(0, 1))
            mm_scale_data = mm_scale.fit_transform(new_close_col)
            new_close_col_val = new_close_col[-60:].values
            new_close_col_val_scale = mm_scale.transform(new_close_col_val)

            X_test = []
            X_test.append(new_close_col_val_scale)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            network = load_model("trained_models/HDFC.model")

            new_preds = network.predict(X_test)
            new_preds = mm_scale.inverse_transform(new_preds)

            NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

            st.subheader(
                "Predictions for the next day Close Price : " + str(NextDay_Date))
            formatter = '{:.2f}'.format(new_preds[0][0])
            st.markdown(html_temp.format(formatter),unsafe_allow_html=True)

            st.subheader("Close Price VS Date Interactive chart for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(data=df1['Close'], width=600, height=300,
                          use_container_width=True)
            
            st.subheader("Line chart of Open and Close for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(df1[['Open', 'Close']])

            st.subheader("Line chart of High and Low for analysis:")
            st.spinner(text='In Progress...')
            st.line_chart(df1[['High', 'Low']])

        if(choose_stock == "ITC"):

            st.title(choose_stock)
            df1 = get_history(symbol='ITC', start=date(
                2015, 1, 1), end=date.today())
            df1['Date'] = df1.index

            new_close_col = df1.filter(['Close'])
            mm_scale = MinMaxScaler(feature_range=(0, 1))
            mm_scale_data = mm_scale.fit_transform(new_close_col)
            new_close_col_val = new_close_col[-60:].values
            new_close_col_val_scale = mm_scale.transform(new_close_col_val)

            X_test = []
            X_test.append(new_close_col_val_scale)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            network = load_model("trained_models/ITC.model")

            new_preds = network.predict(X_test)
            new_preds = mm_scale.inverse_transform(new_preds)

            NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

            st.subheader(
                "Predictions for the next day Close Price : " + str(NextDay_Date))
            formatter = '{:.2f}'.format(new_preds[0][0])
            st.markdown(html_temp.format(formatter),unsafe_allow_html=True)

            st.subheader("Close Price VS Date Interactive chart for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(data=df1['Close'], width=600, height=300,
                          use_container_width=True)

            st.subheader("Line chart of Open and Close for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(df1[['Open', 'Close']])

            st.subheader("Line chart of High and Low for analysis:")
            st.spinner(text='In Progress...')
            st.line_chart(df1[['High', 'Low']])

        if(choose_stock == "NESTLE"):

            st.title(choose_stock)
            df1 = get_history(symbol='NESTLEIND', start=date(
                2015, 1, 1), end=date.today())
            df1['Date'] = df1.index

            new_close_col = df1.filter(['Close'])
            mm_scale = MinMaxScaler(feature_range=(0, 1))
            mm_scale_data = mm_scale.fit_transform(new_close_col)
            new_close_col_val = new_close_col[-60:].values
            new_close_col_val_scale = mm_scale.transform(new_close_col_val)

            X_test = []
            X_test.append(new_close_col_val_scale)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            network = load_model("trained_models/NESTLEIND.model")

            new_preds = network.predict(X_test)
            new_preds = mm_scale.inverse_transform(new_preds)

            NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

            st.subheader(
                "Predictions for the next day Close Price : " + str(NextDay_Date))
            formatter = '{:.2f}'.format(new_preds[0][0])
            st.markdown(html_temp.format(formatter),unsafe_allow_html=True)

            st.subheader("Close Price VS Date Interactive chart for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(data=df1['Close'], width=600, height=300,
                          use_container_width=True)

            st.subheader("Line chart of Open and Close for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(df1[['Open', 'Close']])

            st.subheader("Line chart of High and Low for analysis:")
            st.spinner(text='In Progress...')
            st.line_chart(df1[['High', 'Low']])

        if(choose_stock == "ONGC"):

            st.title(choose_stock)
            df1 = get_history(symbol='ONGC', start=date(
                2015, 1, 1), end=date.today())
            df1['Date'] = df1.index

            new_close_col = df1.filter(['Close'])
            mm_scale = MinMaxScaler(feature_range=(0, 1))
            mm_scale_data = mm_scale.fit_transform(new_close_col)
            new_close_col_val = new_close_col[-60:].values
            new_close_col_val_scale = mm_scale.transform(new_close_col_val)

            X_test = []
            X_test.append(new_close_col_val_scale)
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            network = load_model("trained_models/ONGC.model")

            new_preds = network.predict(X_test)
            new_preds = mm_scale.inverse_transform(new_preds)

            NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

            st.subheader(
                "Predictions for the next day Close Price : " + str(NextDay_Date))
            formatter = '{:.2f}'.format(new_preds[0][0])
            st.markdown(html_temp.format(formatter),unsafe_allow_html=True)

            st.subheader("Close Price VS Date Interactive chart for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(data=df1['Close'], width=600, height=300,
                          use_container_width=True)

            st.subheader("Line chart of Open and Close for analysis:")
            st.spinner(text='In Progress...')
            st.area_chart(df1[['Open', 'Close']])

            st.subheader("Line chart of High and Low for analysis:")
            st.spinner(text='In Progress...')
            st.line_chart(df1[['High', 'Low']])

if __name__ == '__main__':
    main()
