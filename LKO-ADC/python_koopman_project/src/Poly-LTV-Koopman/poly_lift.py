import numpy as np
from itertools import combinations_with_replacement

import numpy as np
from itertools import combinations


def polynomial_expansion_td(state: np.ndarray, p: int, delay_time: int) -> np.ndarray:
    """
    对包含时间延迟的整个状态矩阵进行多项式升维。
    该版本严格复现了指定MATLAB脚本的功能。

    Args:
        state (np.ndarray): 状态矩阵，形状为 (state_size * delay_time, num_samples)。
        p (int): 每个时间步要提升到的目标维度。
        delay_time (int): 时间延迟步长。

    Returns:
        np.ndarray: 升维后的状态矩阵，形状为 (p * delay_time, num_samples)。
    """
    # 验证输入状态矩阵的维度是否与delay_time匹配
    if state.shape[0] % delay_time != 0:
        raise ValueError("状态矩阵的总维度必须是delay_time的整数倍")

    state_size = state.shape[0] // delay_time
    num_samples = state.shape[1]

    # 预分配结果矩阵
    high_dim_state = np.zeros((p * delay_time, num_samples))

    # 遍历每一个样本（每一列）
    for k in range(num_samples):
        # 遍历该样本中的每一个时间步切片
        for t in range(delay_time):
            # 定义在原始矩阵和目标矩阵中的行索引范围
            # Python使用0-based索引，所以直接计算即可
            start_row = state_size * t
            end_row = state_size * (t + 1)

            hd_start_row = p * t
            hd_end_row = p * (t + 1)

            # 提取当前时间步的状态切片
            current_state_slice = state[start_row:end_row, k]

            # 调用与MATLAB功能一致的升维函数
            lifted_slice = adaptive_poly_lift(current_state_slice, p)

            # 将升维后的向量放入结果矩阵的对应位置
            high_dim_state[hd_start_row:hd_end_row, k] = lifted_slice

    return high_dim_state


def adaptive_poly_lift(x: np.ndarray, target_dim: int) -> np.ndarray:
    """
    将单个状态向量x通过多项式扩展提升到目标维度。
    此版本严格复现了MATLAB中使用`nchoosek`的逻辑，
    即只生成不同变量间的乘积项（如x1*x2），不生成自身的幂次项（如x1^2）。

    Args:
        x (np.ndarray): 1维状态向量，形状为 (m,)。
        target_dim (int): 目标升维后的维度。

    Returns:
        np.ndarray: 升维后的向量，形状为 (target_dim,)。
    """
    m = x.shape[0]
    if target_dim < m:
        raise ValueError("目标维度必须大于或等于原始状态维度")

    # 使用列表来高效收集所有项，最后再转换为numpy数组
    lifted_terms = list(x)

    current_dim = m
    current_order = 2  # 从二阶开始

    # 循环生成更高阶的项，直到达到目标维度或阶数上限
    while current_dim < target_dim and current_order <= 5:
        # 使用itertools.combinations复现MATLAB的nchoosek功能
        # 它生成不重复元素的组合，这对应了MATLAB脚本的核心行为
        indices_generator = combinations(range(m), current_order)

        for indices in indices_generator:
            if current_dim >= target_dim:
                break

            # 计算多项式项的值（不同维度状态的乘积）
            term = np.prod(x[list(indices)])
            lifted_terms.append(term)
            current_dim += 1

        current_order += 1

    # 将列表转换为numpy数组并确保其维度正确
    final_lifted_vec = np.array(lifted_terms)

    # 截取到目标维度，以防万一生成的项超过了所需数量
    return final_lifted_vec[:target_dim]

# def adaptive_poly_lift(x: np.ndarray, target_dim: int) -> np.ndarray:
#     """
#     将单个状态向量x通过多项式扩展提升到目标维度。
#     此版本生成了包含自身乘积项的完整多项式基。
#
#     Args:
#         x (np.ndarray): 1维状态向量，形状为 (m,)。
#         target_dim (int): 目标升维后的维度。
#
#     Returns:
#         np.ndarray: 升维后的向量，形状为 (target_dim,)。
#     """
#     m = x.shape[0]
#     if target_dim < m:
#         raise ValueError("目标维度必须大于或等于原始状态维度")
#
#     # 列表用于高效地收集所有项
#     lifted_terms = [x]
#
#     current_dim = m
#     if current_dim >= target_dim:
#         return x[:target_dim]
#
#     # 从二阶多项式开始
#     order = 2
#     # 设置一个最大阶数以防止无限循环
#     while current_dim < target_dim and order <= 5:
#         # 生成当前阶数的所有项的索引组合（包含重复，例如 x_i^2）
#         # indices for [x1*x1, x1*x2, x1*x3, x2*x2, x2*x3, x3*x3] for order=2, m=3
#         term_indices = combinations_with_replacement(range(m), order)
#
#         for indices in term_indices:
#             if current_dim >= target_dim:
#                 break
#
#             # 计算多项式项的值
#             term = np.prod(x[list(indices)])
#             lifted_terms.append(np.array([term]))
#             current_dim += 1
#
#         order += 1
#
#     # 将所有项合并成一个向量并截取到目标维度
#     final_lifted_vec = np.concatenate(lifted_terms)
#
#     return final_lifted_vec[:target_dim]
#
#
# def polynomial_expansion_td(state: np.ndarray, p: int, delay_time: int) -> np.ndarray:
#     """
#     对包含时间延迟的整个状态矩阵进行多项式升维。
#
#     Args:
#         state (np.ndarray): 状态矩阵，形状为 (state_size * delay_time, num_samples)。
#         p (int): 每个时间步要提升到的目标维度。
#         delay_time (int): 时间延迟步长。
#
#     Returns:
#         np.ndarray: 升维后的状态矩阵，形状为 (p * delay_time, num_samples)。
#     """
#     if state.shape[0] % delay_time != 0:
#         raise ValueError("状态矩阵的总维度必须是delay_time的整数倍")
#
#     state_size = state.shape[0] // delay_time
#     num_samples = state.shape[1]
#
#     # 预分配结果矩阵
#     high_dim_state = np.zeros((p * delay_time, num_samples))
#
#     for k in range(num_samples):
#         for t in range(delay_time):
#             # 提取当前样本、当前时间步的状态
#             start_row = state_size * t
#             end_row = state_size * (t + 1)
#             current_state_slice = state[start_row:end_row, k]
#
#             # 升维
#             lifted_slice = adaptive_poly_lift(current_state_slice, p)
#
#             # 放入结果矩阵的对应位置
#             hd_start_row = p * t
#             hd_end_row = p * (t + 1)
#             high_dim_state[hd_start_row:hd_end_row, k] = lifted_slice
#
#     return high_dim_state