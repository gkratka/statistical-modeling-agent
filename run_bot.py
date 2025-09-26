#!/usr/bin/env python
import os
import sys

# Add paths
sys.path.insert(0, os.path.dirname(__file__))

# Set environment variable for token if needed
from dotenv import load_dotenv
load_dotenv()

# Import and run the bot directly
from src.bot.telegram_bot import StatisticalModelingBot

if __name__ == "__main__":
    bot = StatisticalModelingBot()
    bot.run()