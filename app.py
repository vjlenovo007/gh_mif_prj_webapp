from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pypfopt import EfficientFrontier, objective_functions
from pypfopt.black_litterman import BlackLittermanModel

app = Flask(__name__)

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url, header=0)
    tickers = tables[0]['Symbol'].str.replace('.', '-').tolist()
    return tickers

def get_data(tickers, start_date, end_date):
    data = yf.download(" ".join(tickers),
                       start=start_date,
                       end=end_date,
                       interval="1mo",
                       auto_adjust=True)
    prices = data["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    # drop tickers with >30% missing
    valid = prices.columns[prices.isnull().mean() < 0.3]
    return prices[valid].ffill().dropna(how="all")

def compute_returns(prices):
    r = prices.pct_change().dropna()
    return r.replace([np.inf, -np.inf], np.nan).dropna()

def get_market_caps(tickers):
    caps = {}
    for tk in tickers:
        info = yf.Ticker(tk).info
        caps[tk] = (info.get("marketCap", 1e9) / 1e9)
    return caps

def run_black_litterman(returns, caps, views=None, confs=None,
                        delta=2.5, rf=0.0, max_w=0.4):
    # Covariance
    S = returns.cov() * 12
    # Market weights
    mkt_w = pd.Series(caps).loc[returns.columns]
    mkt_w /= mkt_w.sum()
    # Implied returns
    pi = delta * S.dot(mkt_w)
    pi += rf
    # Choose equilibrium or BL
    if views:
        view_s = pd.Series(views)
        bl = BlackLittermanModel(S, pi=pi,
                                 absolute_views=view_s,
                                 view_confidences=confs)
        ret = bl.bl_returns()
    else:
        ret = pi
    # optimize
    ef = EfficientFrontier(ret, S)
    ef.add_constraint(lambda w: w >= 0)
    ef.add_constraint(lambda w: w <= max_w)
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = ef.max_sharpe(rf)
    w_clean = ef.clean_weights(cutoff=0.01)
    perf = ef.portfolio_performance(rf)
    return w_clean, perf

@app.route('/', methods=['GET', 'POST'])
def index():
    tickers = get_sp500_tickers()
    results = None

    if request.method == 'POST':
        # form inputs
        sel = request.form.getlist('tickers')
        start = request.form['start_date']
        end   = request.form['end_date']
        nviews = int(request.form.get('num_views', 0))

        views = {}
        confs = []
        for i in range(nviews):
            t = request.form[f'stock_{i}']
            er = float(request.form[f'expected_{i}'])/100
            cf = float(request.form[f'conf_{i}'])/100
            views[t] = er
            confs.append(cf)

        prices = get_data(sel, start, end)
        rets   = compute_returns(prices)
        caps   = get_market_caps(sel)

        w, perf = run_black_litterman(rets, caps,
                                      views=views or None,
                                      confs=confs or None)

        # Efficient frontier plot (simplified example)
        from plotly.offline import plot
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(w.keys()),
            y=list(w.values()),
            marker_color='steelblue'
        ))
        ef_div = plot(fig, output_type='div', include_plotlyjs=False)

        results = {
            'weights': w,
            'perf': perf,
            'ef_plot': ef_div
        }

    return render_template('index.html',
                           tickers=tickers,
                           results=results)

if __name__ == '__main__':
    app.run(debug=True)
