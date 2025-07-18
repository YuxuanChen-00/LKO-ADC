import torch
def calculate_rmse(y_pred, y_true):
    """
    使用PyTorch计算均方根误差 (RMSE)。

    Args:
        y_pred (torch.Tensor): 模型的预测值。
        y_true (torch.Tensor): 真实值。

    Returns:
        torch.Tensor: 一个包含RMSE值的标量张量。
    """
    # return np.sqrt(((y_pred - y_true) ** 2).mean())
    return torch.sqrt(((y_pred - y_true) ** 2).mean())