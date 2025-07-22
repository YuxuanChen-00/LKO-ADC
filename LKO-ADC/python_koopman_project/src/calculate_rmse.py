import torch
import numpy as np

def calculate_rmse(y_pred, y_true):
    """
    根据输入的数据类型（PyTorch Tensor或NumPy Array），计算均方根误差 (RMSE)。

    Args:
        y_pred (torch.Tensor or np.ndarray): 模型的预测值。
        y_true (torch.Tensor or np.ndarray): 真实值。

    Returns:
        torch.Tensor or np.float64:
        返回RMSE值。如果输入是Tensor，则返回一个标量Tensor；
        如果输入是ndarray，则返回一个NumPy浮点数。

    Raises:
        TypeError: 如果y_pred和y_true的类型不一致，或者不是支持的类型（Tensor或ndarray）。
    """
    # 确保y_pred和y_true的数据类型相同
    if not isinstance(y_pred, type(y_true)):
        raise TypeError(f"输入类型不匹配: y_pred是 {type(y_pred)}, 而 y_true是 {type(y_true)}")

    # 判断数据类型并执行相应的计算
    if isinstance(y_pred, torch.Tensor):
        # 使用PyTorch进行计算
        return torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    elif isinstance(y_pred, np.ndarray):
        # 使用NumPy进行计算
        return np.sqrt(np.mean((y_pred - y_true) ** 2))
    else:
        # 如果类型不支持，则引发错误
        raise TypeError(f"不支持的输入类型: {type(y_pred)}")