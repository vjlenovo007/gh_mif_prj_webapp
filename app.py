from flask import Flask, render_template, request
import io
import base64
import yfinance as yf
import pandas as pd
from pandas.errors import ImportError as PandasImportError
from pypfopt import BlackLittermanModel, risk_models, EfficientFrontier
from pypfopt import plotting
import matplotlib.pyplot as plt

app = Flask(__name__)

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        tables = pd.read_html(url, header=0)
    except PandasImportError:
        raise RuntimeError(
            "Pandas HTML parsing requires 'lxml'. "
            "Please install with `pip install lxml`."
        )
    df = tables[0]
    return df["Symbol"].tolist()

# Default list of choices (fallback)
ALL_TICKERS = get_sp500_tickers()[:50]  # first 50 if you like

@app.errorhandler(500)
def internal_error(exc):
    app.logger.exception(exc)
    return f"<h1>Internal Server Error</h1><pre>{exc}</pre>", 500

@app.route("/", methods=["GET", "POST"])
def index():
    img_data = None
    results = None

    if request.method == "POST":
        tickers = request.form.getlist("tickers") or ALL_TICKERS[:5]

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

        returns = data.pct_change().dropna()

        # 2. Covariance
        S = risk_models.sample_cov(returns)

        # 3. Market caps for pi
        caps = {t: yf.Ticker(t).info.get("marketCap", 0) for t in tickers}
        market_caps = pd.Series(caps)

        # 4. Zero views
        views = pd.Series(0.0, index=returns.columns)
        bl = BlackLittermanModel(S, market_caps=market_caps, absolute_views=views)
        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()

        # 5. Optimize
        ef = EfficientFrontier(ret_bl, cov_bl)
        try:
            raw_weights = ef.max_sharpe(risk_free_rate=0.0)
        except ValueError:
            ef = EfficientFrontier(ret_bl, cov_bl)
            raw_weights = ef.min_volatility()

        exp_ret, vol, sharpe = ef.portfolio_performance(verbose=False)

        # turn into nicely formatted strings
        weights_pct = {t: f"{w*100:.2f}%" for t, w in raw_weights.items()}

        results = {
            "weights": weights_pct,
            "performance": {
                "Expected return": f"{exp_ret:.2%}",
                "Volatility": f"{vol:.2%}",
                "Sharpe ratio": f"{sharpe:.2f}",
            },
        }

        # 6. Plot efficient frontier
        ef_plot = EfficientFrontier(ret_bl, cov_bl)
        fig, ax = plt.subplots()
        plotting.plot_efficient_frontier(ef_plot, ax=ax, show_assets=False)
        ax.set_title("Efficient Frontier (Black–Litterman)")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode("utf8")
        plt.close(fig)

    return render_template(
        "index.html",
        all_tickers=ALL_TICKERS,
        results=results,
        img_data=img_data,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
