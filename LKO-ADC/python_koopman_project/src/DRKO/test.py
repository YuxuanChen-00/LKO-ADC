# 修改后的导入和初始化
import optuna
from optuna.samplers import BOHB

# 创建采样器
sampler = BOHB(seed=123)

# 创建 study
study = optuna.create_study(
    study_name="LKO_BOHB_search",
    direction="minimize",
    sampler=sampler,  # 使用 BOHB 采样器
    storage="sqlite:///LKO_BOHB_search.db"
)