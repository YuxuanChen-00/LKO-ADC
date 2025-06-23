import numpy as np
import torch


def normalize_data(data, params=None):
    """
    使用均值和标准差 (z-score) 对数据进行归一化。
    如果未提供 params，则从数据中计算它们。
    (此函数保持不变)
    """
    if params is None:
        mu = np.mean(data, axis=1, keepdims=True)
        sigma = np.std(data, axis=1, keepdims=True)
        # 避免对常量特征进行零除
        sigma[sigma == 0] = 1
        params = {'mu': mu, 'sigma': sigma}

    normalized_data = (data - params['mu']) / params['sigma']
    return normalized_data, params


def denormalize_data(data, params):
    """
    使用提供的均值和sigma逆转归一化。
    此函数现在可以处理 NumPy 数组和 PyTorch Tensor。
    """
    # 首先检查输入数据的类型
    if isinstance(data, torch.Tensor):
        # 如果是 PyTorch Tensor:
        # 1. 将 numpy 格式的均值和标准差转换为 Tensor
        mu = torch.from_numpy(params['mu'])
        sigma = torch.from_numpy(params['sigma'])

        # 2. 确保 mu 和 sigma 与输入 data 在同一个设备上 (例如 a. 'cpu' 或 b. 'cuda:0')
        mu = mu.to(data.device)
        sigma = sigma.to(data.device)

        # 3. 确保数据类型一致 (例如 torch.float32)
        mu = mu.to(data.dtype)
        sigma = sigma.to(data.dtype)

        # 4. 执行反归一化操作
        return data * sigma + mu

    elif isinstance(data, np.ndarray):
        # 如果是 NumPy Array，执行原始逻辑
        return data * params['sigma'] + params['mu']

    else:
        # 处理未知类型
        raise TypeError(f"不支持的数据类型: {type(data)}. 函数只接受 numpy.ndarray 或 torch.Tensor。")