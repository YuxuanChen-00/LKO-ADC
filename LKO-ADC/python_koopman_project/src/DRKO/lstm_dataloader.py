import torch
from torch.utils.data import Dataset, DataLoader


class CustomTimeSeriesDataset(Dataset):
    """
    用于时间序列数据的自定义PyTorch数据集。
    它将字典形式的numpy数组数据封装成PyTorch可用的数据集对象。
    """

    def __init__(self, data_dict):
        """
        初始化数据集。

        Args:
            data_dict (dict): 包含 'control', 'state', 'label' NumPy数组的数据字典。
        """
        # 将Numpy数组转换为PyTorch张量
        self.state = torch.from_numpy(data_dict['state']).float()
        self.control = torch.from_numpy(data_dict['control']).float()
        self.label = torch.from_numpy(data_dict['label']).float()

        # 确保所有数据样本数量一致
        self.num_samples = self.state.shape[0]
        assert self.control.shape[0] == self.num_samples, "Control data size mismatch"
        assert self.label.shape[0] == self.num_samples, "Label data size mismatch"

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        根据索引 idx 获取单个数据样本。

        Args:
            idx (int): 数据样本的索引。

        Returns:
            tuple: 包含 (state, control, label) 的元组。
        """
        return self.state[idx], self.control[idx], self.label[idx]


