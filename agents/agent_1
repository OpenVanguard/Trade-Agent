import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class SMACrossoverAgent:
    def __init__(self, symbol, short_window=40, long_window=100):
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.data = None

    def fetch_data(self, start_date, end_date):
        """Fetch historical stock data for the given symbol."""
        self.data = yf.download(self.symbol, start=start_date, end=end_date)
        self.data['Short_MA'] = self.data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        self.data['Long_MA'] = self.data['Close'].rolling(window=self.long_window, min_periods=1).mean()

    def generate_signals(self):
        """Generate buy/sell signals based on SMA crossover."""
        self.data['Signal'] = 0  # No action
        self.data['Signal'][self.short_window:] = \
            (self.data['Short_MA'][self.short_window:] > self.data['Long_MA'][self.short_window:]).astype(int)
        self.data['Position'] = self.data['Signal'].diff()

    def plot_data(self):
        """Visualize the stock price along with short and long moving averages and buy/sell signals."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['Close'], label='Close Price')
        plt.plot(self.data['Short_MA'], label=f'{self.short_window}-Day SMA', alpha=0.7)
        plt.plot(self.data['Long_MA'], label=f'{self.long_window}-Day SMA', alpha=0.7)
        
        plt.plot(self.data.loc[self.data['Position'] == 1].index,
                 self.data['Short_MA'][self.data['Position'] == 1],
                 '^', markersize=10, color='g', lw=0, label='Buy Signal')
        plt.plot(self.data.loc[self.data['Position'] == -1].index,
                 self.data['Short_MA'][self.data['Position'] == -1],
                 'v', markersize=10, color='r', lw=0, label='Sell Signal')
        
        plt.title(f'{self.symbol} SMA Crossover Strategy')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def run(self, start_date, end_date):
        """Execute the agent's workflow: fetch data, generate signals, and plot."""
        self.fetch_data(start_date, end_date)
        self.generate_signals()
        self.plot_data()


# Parameters
symbol = 'AAPL'
start_date = '2022-01-01'
end_date = '2023-01-01'
short_window = 40
long_window = 100

# Initialize and run the agent
agent = SMACrossoverAgent(symbol=symbol, short_window=short_window, long_window=long_window)
agent.run(start_date, end_date)
