import numpy as np
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import warnings

# 导入您提供的多项式升维函数
try:
    from poly_lift import polynomial_expansion_td
except ImportError:
    print("错误：无法导入 'poly_lift.py'。请确保该文件与本脚本在同一目录下。")
    exit()

warnings.filterwarnings('ignore', message='Ill-conditioned matrix')


class KoopmanModel_ABC:
    """
    最终稳定版:
    - Koopman算子分解为 A, B, C 形式
    - 包含正则化项提高数值稳定性
    - 预测循环在高维空间中进行以提高效率
    - 绘图部分修改为保存文件以兼容所有环境
    """

    def __init__(self, delay_time: int, p_dim: int, is_norm: bool = True, alpha: float = 1e-5):
        self.delay_time = delay_time
        self.p_dim = p_dim
        self.is_norm = is_norm
        self.alpha = alpha

        self.A, self.B, self.C = None, None, None
        self.state_scaler, self.input_scaler = None, None
        self.state_dim, self.input_dim, self.lifted_dim = None, None, None

        print(f"模型初始化 (最终稳定版): delay_time={delay_time}, p_dim={p_dim}, is_norm={is_norm}, alpha={alpha}")

    def _load_mat_data(self, data_path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
        # (无修改)
        states_list = []
        inputs_list = []
        if not data_path.is_dir():
            print(f"错误: 路径 {data_path} 不是一个有效的目录。")
            return [], []
        mat_files = sorted(list(data_path.glob("*.mat")))
        print(f"在 {data_path} 中找到 {len(mat_files)} 个 .mat 文件。")
        for file in mat_files:
            try:
                data = scipy.io.loadmat(file)
                if 'state' in data and 'input' in data:
                    states_list.append(data['state'])
                    inputs_list.append(data['input'])
                else:
                    print(f"警告: 文件 {file} 中缺少 'state' 或 'input' 变量。")
            except Exception as e:
                print(f"加载文件 {file} 时出错: {e}")
        return states_list, inputs_list

    def _create_time_delay_data(self, states: np.ndarray, inputs: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        # (无修改)
        num_samples = states.shape[1]
        if num_samples < self.delay_time + 1:
            return np.array([]), np.array([]), np.array([])
        X_delay, Y_target, U_input = [], [], []
        for i in range(self.delay_time - 1, num_samples - 1):
            delay_vec = states[:, i - self.delay_time + 1: i + 1].T.flatten()
            X_delay.append(delay_vec)
            Y_target.append(states[:, i + 1])
            U_input.append(inputs[:, i])
        return np.array(X_delay).T, np.array(Y_target).T, np.array(U_input).T

    def fit(self, train_path: Path):
        # (无修改)
        print("\n--- 开始训练过程 (A,B,C分解+正则化) ---")
        train_states_list, train_inputs_list = self._load_mat_data(train_path)
        if not train_states_list: return
        self.state_dim = train_states_list[0].shape[0]
        self.input_dim = train_inputs_list[0].shape[0]
        self.lifted_dim = self.p_dim * self.delay_time
        if self.is_norm:
            print("正在计算归一化参数并对数据进行归一化...")
            all_train_states = np.hstack(train_states_list)
            all_train_inputs = np.hstack(train_inputs_list)
            self.state_scaler = {'mean': np.mean(all_train_states, axis=1, keepdims=True),
                                 'std': np.std(all_train_states, axis=1, keepdims=True)}
            self.input_scaler = {'mean': np.mean(all_train_inputs, axis=1, keepdims=True),
                                 'std': np.std(all_train_inputs, axis=1, keepdims=True)}
            self.state_scaler['std'][self.state_scaler['std'] == 0] = 1
            self.input_scaler['std'][self.input_scaler['std'] == 0] = 1
            for i in range(len(train_states_list)):
                train_states_list[i] = (train_states_list[i] - self.state_scaler['mean']) / self.state_scaler['std']
                train_inputs_list[i] = (train_inputs_list[i] - self.input_scaler['mean']) / self.input_scaler['std']
        print("正在生成时间延迟矩阵并升维...")
        X_delay_all, U_all = [], []
        for states, inputs in zip(train_states_list, train_inputs_list):
            X_d, _, U_i = self._create_time_delay_data(states, inputs)
            if X_d.size > 0:
                X_delay_all.append(X_d)
                U_all.append(U_i)
        X_delay_matrix = np.hstack(X_delay_all)
        U_matrix = np.hstack(U_all)
        Z = polynomial_expansion_td(X_delay_matrix, self.p_dim, self.delay_time)
        print(f"总训练样本数: {Z.shape[1]}")
        Z_current = Z[:, :-1]
        Z_next = Z[:, 1:]
        U_current = U_matrix[:, :-1]
        print("正在使用岭回归计算 A 和 B 矩阵...")
        Omega = np.vstack([Z_current, U_current])
        I = np.eye(Omega.shape[0])
        K_lifted = Z_next @ Omega.T @ np.linalg.inv(Omega @ Omega.T + self.alpha * I)
        self.A = K_lifted[:, :self.lifted_dim]
        self.B = K_lifted[:, self.lifted_dim:]
        print("正在构建解码矩阵 C...")
        self.C = np.zeros((self.state_dim, self.lifted_dim))
        start_col = self.p_dim * (self.delay_time - 1)
        end_col = start_col + self.state_dim
        self.C[:, start_col:end_col] = np.eye(self.state_dim)
        print("训练完成。")
        print(f"  - 矩阵 A 的形状: {self.A.shape}")
        print(f"  - 矩阵 B 的形状: {self.B.shape}")
        print(f"  - 矩阵 C 的形状: {self.C.shape}")

    def predict(self, test_path: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        # (无修改)
        if self.A is None or self.B is None or self.C is None:
            raise RuntimeError("模型尚未训练，请先调用 .fit() 方法。")
        print("\n--- 开始预测过程 (高维空间迭代版) ---")
        test_states_list, test_inputs_list = self._load_mat_data(test_path)
        if not test_states_list: return [], [], []
        predictions = []
        ground_truths = []
        for i, (states, inputs) in enumerate(zip(test_states_list, test_inputs_list)):
            print(f"正在处理测试轨迹 {i + 1}/{len(test_states_list)}...")
            ground_truths.append(states.copy())
            if self.is_norm:
                states = (states - self.state_scaler['mean']) / self.state_scaler['std']
                inputs = (inputs - self.input_scaler['mean']) / self.input_scaler['std']
            T = states.shape[1]
            if T < self.delay_time:
                print(f"  警告: 轨迹 {i + 1} 长度 ({T}) 小于延迟时间 ({self.delay_time})，无法预测。")
                predictions.append(np.full_like(ground_truths[-1], np.nan))
                continue
            initial_history = states[:, :self.delay_time]
            initial_delay_vec = initial_history.T.flatten().reshape(-1, 1)
            z_current = polynomial_expansion_td(initial_delay_vec, self.p_dim, self.delay_time)
            predicted_z_sequence = []
            for k in range(self.delay_time, T):
                u_k_minus_1 = inputs[:, k - 1].reshape(-1, 1)
                z_next = self.A @ z_current + self.B @ u_k_minus_1
                if not np.all(np.isfinite(z_next)):
                    print(f"  警告: 在轨迹 {i + 1} 的第 {k} 步预测中出现数值不稳定 (inf or NaN)。")
                    remaining_steps = T - k
                    nan_z_padding = np.full((self.lifted_dim, remaining_steps), np.nan)
                    if predicted_z_sequence:
                        predicted_z_sequence.append(nan_z_padding)
                    else:
                        predicted_z_sequence = [nan_z_padding]
                    break
                predicted_z_sequence.append(z_next)
                z_current = z_next
            if not predicted_z_sequence:
                z_matrix = np.empty((self.lifted_dim, 0))
            else:
                try:
                    z_matrix = np.hstack(predicted_z_sequence)
                except ValueError:
                    first_nan_idx = next(
                        (idx for idx, z in enumerate(predicted_z_sequence) if z.ndim > 1 and np.isnan(z).any()), None)
                    if first_nan_idx is not None:
                        z_matrix = np.hstack(predicted_z_sequence[:first_nan_idx])
                    else:
                        z_matrix = np.empty((self.lifted_dim, 0))
            predicted_future_states = self.C @ z_matrix
            predicted_states = np.full_like(states, np.nan)
            predicted_states[:, :self.delay_time] = initial_history
            num_predicted_steps = predicted_future_states.shape[1]
            end_idx = self.delay_time + num_predicted_steps
            predicted_states[:, self.delay_time:end_idx] = predicted_future_states
            if self.is_norm:
                predicted_states = predicted_states * self.state_scaler['std'] + self.state_scaler['mean']
            predictions.append(predicted_states)
        return predictions, ground_truths, test_inputs_list

    def evaluate_and_plot(self, predictions: list, ground_truths: list):
        # *** 此处为核心修正 ***
        print("\n--- 开始评估和绘图 ---")
        if not predictions:
            print("没有可供评估的预测结果。")
            return
        all_rmse = []
        for i, (pred, true) in enumerate(zip(predictions, ground_truths)):
            valid_indices = ~np.isnan(pred).any(axis=0)
            if not np.any(valid_indices):
                print(f"轨迹 {i + 1} 的所有预测值均为NaN，无法计算RMSE。")
                all_rmse.append(np.inf)
                continue

            rmse = np.sqrt(mean_squared_error(true[:, valid_indices].T, pred[:, valid_indices].T))
            all_rmse.append(rmse)
            print(f"轨迹 {i + 1} 的 RMSE: {rmse:.4f}")

        valid_rmse = [r for r in all_rmse if np.isfinite(r)]
        if valid_rmse:
            avg_rmse = np.mean(valid_rmse)
            print(f"\n所有有效轨迹的平均 RMSE: {avg_rmse:.4f}")
        else:
            print("\n所有轨迹均未能成功预测，无法计算平均RMSE。")

        # 绘图部分修改
        num_trajectories = len(predictions)
        for i in range(num_trajectories):
            pred, true = predictions[i], ground_truths[i]
            valid_indices = ~np.isnan(pred).any(axis=0)

            T, state_dims = pred.shape[1], self.state_dim
            time_axis = np.arange(T)

            # 为每个轨迹创建一个新的图像对象
            fig, axes = plt.subplots(state_dims, 1, figsize=(15, 2 * state_dims), sharex=True, squeeze=False)
            axes = axes.flatten()  # 确保axes总是一个可迭代的数组

            fig.suptitle(f'Trajectory {i + 1} Prediction vs Ground Truth (RMSE: {all_rmse[i]:.4f})', fontsize=16)

            for j in range(state_dims):
                axes[j].plot(time_axis, true[j, :], 'b-', label='Ground Truth')
                axes[j].plot(time_axis[valid_indices], pred[j, valid_indices], 'r--', label='Prediction')
                axes[j].set_ylabel(f'State dim {j + 1}')
                axes[j].legend()
                axes[j].grid(True)

            axes[-1].set_xlabel('Time Step')
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # 将 plt.show() 替换为 plt.savefig()
            plot_filename = f"trajectory_{i + 1}_prediction.png"
            plt.savefig(plot_filename)
            print(f"  - 图像已保存至: {plot_filename}")

            # 关闭图像以释放内存，这在循环中绘图时是个好习惯
            plt.close(fig)


if __name__ == '__main__':
    # --- 参数设置 ---
    DELAY_TIME = 5
    P_DIM = 25
    IS_NORM = True
    ALPHA = 1e-5  # 正则化超参数

    # --- 路径设置 ---
    try:
        current_dir = Path(__file__).resolve().parent
        # 路径根据您的描述进行设置
        base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
        train_path = base_data_path / "80minTrain"
        test_path = base_data_path / "50secTest"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError
    except (FileNotFoundError, NameError):
        print("错误: 默认数据路径未找到。")
        print("请根据您的文件结构手动设置 'train_path' 和 'test_path' 到您的数据目录。")
        exit()

    # 1. 实例化模型
    model = KoopmanModel_ABC(delay_time=DELAY_TIME, p_dim=P_DIM, is_norm=IS_NORM, alpha=ALPHA)

    # 2. 训练模型
    model.fit(train_path)

    # 3. 进行预测
    predictions, ground_truths, test_inputs = model.predict(test_path)

    # 4. 评估和绘图
    if predictions:
        model.evaluate_and_plot(predictions, ground_truths)