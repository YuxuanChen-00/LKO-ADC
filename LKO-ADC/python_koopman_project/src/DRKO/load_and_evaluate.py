import torch
import numpy as np
from scipy.io import loadmat
import matplotlib

matplotlib.use('Agg')  # 或者 'TkAgg', 'Qt5Agg' 等，Agg 是非交互式后端，适合在无图形界面服务器上运行
import matplotlib.pyplot as plt
from pathlib import Path

# --- Import the functions and classes we previously converted ---
# These should be in your project directory

from evaluate_lstm_lko import evaluate_lstm_lko, evaluate_lstm_lko2
from generate_lstm_data import generate_lstm_data
from src.normalize_data import normalize_data, denormalize_data
from train_lstm_lko import train_lstm_lko
from evaluate_lstm_lko import calculate_rmse