import torch
import numpy as np
from scipy.io import loadmat
import random
from pathlib import Path
import time
import os
import ray
from ray import tune
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
import ConfigSpace as CS
import pandas as pd
from ray.train import Checkpoint
# ==================== 核心修正: 更改Trial的导入路径 ====================
from ray.tune.experiment.trial import Trial
# ===================================================================

# --- 导入您项目中的核心模块 ---
from train_hybrid_ltv_model import train_two_stage_hybrid_model
from dataloader import generate_koopman_data
from src.normalize_data import normalize_data

# --- 全局路径设置 (保持不变) ---
BASE_MODEL_SAVE_PATH = Path(__file__).resolve().parent / "models" / "BOHB_Search_TwoStage_Final"


# 1. 定义 Ray Tune 的“可训练”函数 (保持不变)
def bohb_train_trial(config, raw_data_dict, base_params):
    """
    这个函数是Ray Tune为每一次超参数组合调用的执行单元。
    """
    params = base_params.copy()
    params.update(config)
    params['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for key in ['encoder_gru_hidden', 'encoder_mlp_hidden', 'delta_rnn_hidden', 'delta_mlp_hidden',
                'num_epochs_s1', 'num_epochs_s2', 'pred_step', 'delay_step', 'i_multiplier']:
        if key in params:
            params[key] = int(params[key])

    params['PhiDimensions'] = params['delay_step'] * params['i_multiplier']

    # --- 数据生成 ---
    def process_data_files(file_path_list, params):
        data_list = []
        for file_path in file_path_list:
            data = loadmat(file_path)
            state_data, _ = normalize_data(data['state'], params['params_state'])
            ctrl_td, state_td, label_td = generate_koopman_data(data['input'], state_data, params['delay_step'],
                                                                params['pred_step'])
            data_list.append({
                'control': torch.from_numpy(ctrl_td).float(),
                'state': torch.from_numpy(state_td).float(),
                'label': torch.from_numpy(label_td).float()
            })
        return data_list

    raw_train_paths = raw_data_dict['raw_train_paths']
    raw_test_paths = raw_data_dict['raw_test_paths']

    train_data_list = process_data_files(raw_train_paths, params)
    train_data = {'control': np.concatenate([d['control'].numpy() for d in train_data_list], axis=0),
                  'state': np.concatenate([d['state'].numpy() for d in train_data_list], axis=0),
                  'label': np.concatenate([d['label'].numpy() for d in train_data_list], axis=0)}
    test_data = process_data_files(raw_test_paths, params)

    # --- 调用训练流程 ---
    final_net, best_rmse = train_two_stage_hybrid_model(params, train_data, test_data)

    # --- 保存检查点 ---
    checkpoint = None
    if final_net is not None:
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(final_net.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

    metrics_to_report = {"rmse": best_rmse}
    tune.report(metrics_to_report, checkpoint=checkpoint)

# 定义一个简短的试验名称生成函数
def short_trial_name_creator(trial: Trial) -> str:
    """
    使用试验的唯一ID来创建一个简短的目录名称。
    """
    return f"trial_{trial.trial_id}"

# 2. 主执行流程
def main():
    start_time_total = time.time()
    print("## 1. 设置基础参数和路径... ##")
    BASE_MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    # ==================== 代码修改点 1 ====================
    # 将 delay_step 和 i_multiplier 添加为固定参数
    base_params = {
        'is_norm': True, 'state_size': 6, 'control_size': 6,
        'batchSize': 1024, 'seed': 666, 'minLearnRate': 1e-7,
        'num_epochs_s1': 100,
        'num_epochs_s2': 50,
        'delay_step': 6,       # 设置为固定值
        'i_multiplier': 20     # 设置为固定值
    }
    # =======================================================


    print("\n## 2. 加载原始数据和计算标准化参数... ##")
    current_dir = Path(__file__).resolve().parent
    base_data_path = current_dir.parent.parent / "data" / "SorotokiData" / "MotionData8" / "FilteredDataPos"
    train_path, test_path = base_data_path / "80minTrain", base_data_path / "50secTest"
    train_files, test_files = sorted(list(train_path.glob('*.mat'))), sorted(list(test_path.glob('*.mat')))

    state_for_norm = np.concatenate([loadmat(f)['state'] for f in train_files], axis=1)
    _, params_state = normalize_data(state_for_norm)
    base_params['params_state'] = params_state
    print("归一化参数计算完毕。")

    raw_data_dict = {
        'raw_train_paths': train_files,
        'raw_test_paths': test_files
    }
    print("原始数据路径已收集。数据加载和序列化将在每个试验中动态进行。")

    print("\n## 3. 定义超参数搜索空间... ##")
    config_space = CS.ConfigurationSpace()
    # ==================== 代码修改点 2 ====================
    # 注释掉或删除对 delay_step 和 i_multiplier 的搜索
    # config_space.add(CS.CategoricalHyperparameter("delay_step", [2, 4, 6, 8, 10]))
    # config_space.add(CS.CategoricalHyperparameter("i_multiplier", [12,14,16,18,20,22,24,26,28,30]))
    # =======================================================
    config_space.add(CS.UniformFloatHyperparameter("L1", 1e-1, 1e1, log=True))
    config_space.add(CS.UniformFloatHyperparameter("L2", 1e-1, 1e1, log=True))
    config_space.add(CS.UniformFloatHyperparameter("L3", 1e1, 1e3, log=True))
    config_space.add(CS.UniformFloatHyperparameter("L_delta", 1e-4, 1.0, log=True))
    config_space.add(CS.UniformFloatHyperparameter("lr_s1", 1e-6, 1e-4, log=True))
    config_space.add(CS.UniformFloatHyperparameter("lr_s2", 1e-7, 1e-5, log=True))
    config_space.add(CS.CategoricalHyperparameter("encoder_gru_hidden", [128, 256, 512]))
    config_space.add(CS.CategoricalHyperparameter("encoder_mlp_hidden", [64, 128, 256]))
    config_space.add(CS.CategoricalHyperparameter("delta_rnn_hidden", [32, 64, 128]))
    config_space.add(CS.CategoricalHyperparameter("delta_mlp_hidden", [32, 64, 128]))
    config_space.add(CS.UniformIntegerHyperparameter("pred_step", 1, 10))

    print("\n## 4. 开始BOHB超参数搜索... ##")
    ray.init(num_gpus=torch.cuda.device_count(), ignore_reinit_error=True)
    bohb_search_alg = TuneBOHB(space=config_space, metric="rmse", mode="min")

    max_epochs = base_params["num_epochs_s1"] + base_params["num_epochs_s2"]
    bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=max_epochs)

    trainable_with_resources = tune.with_resources(bohb_train_trial, resources={"cpu": 1.5, "gpu": 0.1})

    tuner = tune.Tuner(
        tune.with_parameters(trainable_with_resources, raw_data_dict=raw_data_dict, base_params=base_params),
        run_config=tune.RunConfig(name="BOHB_Search_FullArch_Final", storage_path=str(BASE_MODEL_SAVE_PATH)), # 保持不变
        tune_config=tune.TuneConfig(
            scheduler=bohb_scheduler,
            search_alg=bohb_search_alg,
            num_samples=200,
            metric="rmse",
            mode="min",
            trial_name_creator=short_trial_name_creator
        )
    )
    results = tuner.fit()

    print(f"\n\n{'=' * 80}\n所有超参数搜索完成! 总耗时: {(time.time() - start_time_total) / 3600:.2f} 小时\n{'=' * 80}")
    results_df = results.get_dataframe()
    if not results_df.empty:
        print("搜索完成，最优结果如下：")
        best_result = results.get_best_result(metric="rmse", mode="min")
        if best_result:
            print("Best trial config: {}".format(best_result.config))
            if best_result.metrics:
                print("Best trial final validation RMSE: {}".format(best_result.metrics.get("rmse")))
            results_df.to_csv(BASE_MODEL_SAVE_PATH / "full_bohb_search_results.csv") # 保持不变
    else:
        print("未找到任何有效的试验结果。")
    ray.shutdown()


if __name__ == '__main__':
    main()