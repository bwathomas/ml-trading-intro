# Buy Low, Sell High (It's Harder Than It Sounds)

Hands-on lab notebook for an introductory lecture on **machine learning for trading**:
GameStop vs. the S&P 500 vs. Bitcoin (plus any tickers you want to add).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bwathomas/ml-trading-intro/blob/main/ml_trading_intro.ipynb)

## What's inside

| Part | Topic | Lecture tie-in |
|---|---|---|
| 1 | Data exploration with `yfinance`; buy-&-hold vs. $1/day measurement; start-date sensitivity | implicit p-hacking |
| 2 | Online learning on the full S&P 500: random picks, a forward-looking oracle, Follow-The-Leader, Hedge (multiplicative weights); the effect of fees | regularized FTL, regret, fees & slippage |
| 3 | Classical forecasting: AR, MA, ARIMA configs, quarterly S-ARIMA, all evaluated past a train/test cutoff | backtesting honestly |
| 4 | RNN and GRU forecasters vs. an "embarrassingly simple" linear model (Zeng et al., 2022); sign-trading backtest with Sharpe ratios | alpha vs. beta, transformers aren't all you need |

Every strategy is scored the same way: **invest $1 every day, sell at the close**, and
report cumulative profit plus the annualized **Sharpe ratio**.

## Running it

Click the Colab badge above and `Runtime → Run all`. The notebook runs end-to-end in a
few minutes on Colab's free tier (a T4 GPU helps Part 4 but isn't required — all models
are deliberately tiny). The only network dependencies are Yahoo Finance (prices) and
Wikipedia (the S&P 500 member list).

## For students

- Add your own tickers in the first cell of Part 1 — everything downstream adapts.
- Suggested experiments are listed in the final cell (tune `eta`, move the cutoff,
  swap in an LSTM, ...).

> ⚖️ Educational use only. Nothing in this repository is financial advice.
