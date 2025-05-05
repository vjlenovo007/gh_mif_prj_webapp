from flask import Flask, render_template, request
import io
import base64
import yfinance as yf
import pandas as pd
from pypfopt import BlackLittermanModel, risk_models, EfficientFrontier
import matplotlib.pyplot as plt

app = Flask(__name__)

# Default list of companies for the portfolio
ALL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

@app.route("/", methods=["GET", "POST"])
def index():
    img_data = None
    results = None

    if request.method == "POST":
        tickers = request.form.getlist("tickers")

        # 1. Fetch historical data
        raw = yf.download(
            tickers,
            period="1y",
            interval="1wk",
            auto_adjust=False,
        )
        # Extract closing prices
        if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
            data = raw["Close"]
        elif isinstance(raw, pd.Series):
            data = raw.to_frame(name=tickers[0])
        else:
            data = raw

        # Compute returns
        returns = data.pct_change().dropna()

        # 2. Estimate the market covariance
        S = risk_models.sample_cov(returns)

        # 3. Provide zero views for Black-Litterman to avoid Q/P errors
        views = pd.Series(0.0, index=returns.columns)
        bl = BlackLittermanModel(S, absolute_views=views)
        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()

        # 4. Optimize for maximum Sharpe ratio
        ef = EfficientFrontier(ret_bl, cov_bl)
        weights = ef.max_sharpe()
        perf = ef.portfolio_performance(verbose=True)

        results = {
            "weights": weights,
            "performance": {
                "Expected return": perf[0],
                "Volatility": perf[1],
                "Sharpe ratio": perf[2],
            },
        }

        # 5. Plot the efficient frontier
        fig, ax = plt.subplots()
        ef.plot_efficient_frontier(ax=ax, show_assets=False)
        ax.set_title("Efficient Frontier (Black–Litterman)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode("utf8")

    return render_template(
        "index.html",
        all_tickers=ALL_TICKERS,
        results=results,
        img_data=img_data,
    )

if __name__ == "__main__":
    # For local debugging; on Azure App Service this is handled by gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
