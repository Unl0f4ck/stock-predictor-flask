import os
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, send_file
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import plotly.graph_objs as go
import joblib

app = Flask(__name__)
STATIC_DIR = "static"
MODEL_PATH = "stock_dl_model.h5"
SCALER_PATH = "scaler.pkl"

os.makedirs(STATIC_DIR, exist_ok=True)
model = load_model(MODEL_PATH)


def safe_mape(y_true, y_pred):
    mask = (y_true != 0)
    if mask.sum() == 0:
        return np.nan
    return (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock = request.form.get('stock', '').strip().upper()
        if not stock:
            return render_template('index.html', error="⚠️ Please enter a stock symbol (e.g., TCS.NS, INFY.NS, TSLA).")

        # Fetch data
        df = yf.download(stock, start="2015-01-01", end=dt.datetime.now(), interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        if df.empty or 'Close' not in df.columns:
            return render_template('index.html', error=f"❌ No valid data for {stock}")

        df = df.dropna(subset=['Close']).copy()

        # Live stats
        current_price = round(df['Close'].iloc[-1], 2)
        prev_close = round(df['Close'].iloc[-2], 2)
        pct_change = round(((current_price - prev_close) / prev_close) * 100, 2)
        open_price = round(df['Open'].iloc[-1], 2)
        high_price = round(df['High'].iloc[-1], 2)
        low_price = round(df['Low'].iloc[-1], 2)
        volume = int(df['Volume'].iloc[-1])

        # Scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        else:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df[['Close']])
            joblib.dump(scaler, SCALER_PATH)

        # LSTM prep
        data_close = df[['Close']]
        split_idx = int(len(data_close) * 0.7)
        data_training = data_close.iloc[:split_idx]
        data_testing = data_close.iloc[split_idx:]
        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing])
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i - 100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        if len(x_test) == 0:
            return render_template('index.html', error="Not enough data for prediction.")

        # LSTM
        y_pred_scaled = model.predict(x_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = float(np.sqrt(mean_squared_error(y_test_orig, y_pred)))
        mae = float(mean_absolute_error(y_test_orig, y_pred))
        mape = float(safe_mape(y_test_orig, y_pred))

        # Next-day LSTM
        last_100_days = data_close.tail(100).values
        scaled_input = scaler.transform(last_100_days)
        x_input = np.array(scaled_input).reshape(1, 100, 1)
        next_day_scaled = model.predict(x_input)[0][0]
        lstm_pred = float(scaler.inverse_transform(np.array([[next_day_scaled]]))[0][0])

        # Random Forest & Linear Regression
        df['Prev_Close'] = df['Close'].shift(1)
        df['MA5'] = df['Close'].rolling(5).mean()
        df['MA10'] = df['Close'].rolling(10).mean()
        df.dropna(inplace=True)
        X = df[['Prev_Close', 'MA5', 'MA10']]
        y = df['Close']

        rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
        rf_pred = float(rf.predict([X.iloc[-1].values])[0])
        lr = LinearRegression().fit(X, y)
        lr_pred = float(lr.predict([X.iloc[-1].values])[0])

        # EMAs
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()
        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()

        # CHARTS
        # 1. EMA 20 & 50
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=df.index, y=ema20, name='EMA20', line=dict(color='orange')))
        fig1.add_trace(go.Scatter(x=df.index, y=ema50, name='EMA50', line=dict(color='red')))
        fig1.update_layout(title="EMA (20 & 50)", template='plotly_white', height=400)
        chart1_html = fig1.to_html(full_html=False)

        # 2. EMA 100 & 200
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=df.index, y=ema100, name='EMA100', line=dict(color='green')))
        fig2.add_trace(go.Scatter(x=df.index, y=ema200, name='EMA200', line=dict(color='purple')))
        fig2.update_layout(title="EMA (100 & 200)", template='plotly_white', height=400)
        chart2_html = fig2.to_html(full_html=False)

        # 3. LSTM vs Actual
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=y_test_orig.flatten(), name='Actual', line=dict(color='green')))
        fig3.add_trace(go.Scatter(y=y_pred.flatten(), name='Predicted', line=dict(color='red')))
        fig3.update_layout(title="LSTM Prediction vs Actual", template='plotly_white', height=400)
        chart3_html = fig3.to_html(full_html=False)

        # 4. Model Comparison
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=df.index[-100:], y=df['Close'].tail(100), name='Actual', line=dict(color='blue')))
        fig4.add_hline(y=lstm_pred, line_dash="dash", line_color="red", annotation_text=f"LSTM ₹{round(lstm_pred,2)}")
        fig4.add_hline(y=rf_pred, line_dash="dash", line_color="green", annotation_text=f"RF ₹{round(rf_pred,2)}")
        fig4.add_hline(y=lr_pred, line_dash="dash", line_color="orange", annotation_text=f"LR ₹{round(lr_pred,2)}")
        fig4.update_layout(title="Model Comparison (Next-Day Prediction)", template='plotly_white', height=400)
        chart4_html = fig4.to_html(full_html=False)

        # Save CSV
        csv_name = f"{stock}_dataset.csv"
        df.to_csv(os.path.join(STATIC_DIR, csv_name))

        metrics = {"RMSE": round(rmse, 3), "MAE": round(mae, 3), "MAPE%": round(mape, 3)}
        predictions = {"LSTM": lstm_pred, "Random Forest": rf_pred, "Linear Regression": lr_pred}

        return render_template(
            'index.html',
            stock=stock,
            chart1_html=chart1_html,
            chart2_html=chart2_html,
            chart3_html=chart3_html,
            chart4_html=chart4_html,
            current_price=current_price,
            pct_change=pct_change,
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            volume=volume,
            predictions=predictions,
            metrics=metrics,
            dataset_link=csv_name,
            current_time=dt.datetime.now().strftime("%I:%M %p")
        )

    return render_template('index.html')


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(STATIC_DIR, filename), as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
