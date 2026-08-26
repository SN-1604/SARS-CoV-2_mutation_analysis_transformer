# SARS-CoV-2_mutation_analysis_transformer

基于 Transformer 表征与时间序列预测的新冠病毒（SARS-CoV-2）变异速度影响因素分析管线。

本项目以 GISAID 的 spike 核苷酸序列为变异信息源、以 Our World in Data（OWID）疫情指标面板为因素信息源，依次完成：序列月度分组与多序列比对 → 逐位点香农信息熵计算（变异速度代理指标）→ OWID 多维指标 Transformer 自监督表征 → 月度熵时间序列预测 → 基于控制变量扰动的事后解释与"疫苗接种 × 政策管控"组合策略寻优。

## 功能模块

| 模块 | 功能 |
| --- | --- |
| 序列处理 | 从 64.6GB GISAID spike 序列合集筛选、按月归档、去重，构建 219 个国家/地区的月度序列集 |
| 多序列比对 | MAFFT 多线程逐月比对（含超大文件分块比对 + merge 合并策略）与比对质控清洗 |
| 熵计算 | 逐位点香农熵按月求和，5 点居中滑动平均平滑 |
| 表征模型 | 13 周滑动窗口 Transformer 自监督预训练，受控探针选型，生成 64 维月度表征（无前视） |
| 熵预测 | K=3 残差架构 + 3 种子集成，以月度表征 + 历史熵预测下一月平滑熵（验证集 R²≈0.82） |
| 事后解释 | 单因素比例扰动敏感性分析（现实约束 + 辅助模型联动）与三因素组合网格寻优 |

## 目录结构

```
SARS-CoV-2_mutation_analysis_transformer/
├── 序列处理
│   ├── extract_spike_monthly.py      # spike 序列筛选、按国家×月归档、去重、国名合并
│   ├── reorg_country_dirs.py         # 同义国家目录归并整理
│   └── dedupe_files.py               # 月度 fasta 逐文件去重复扫
├── 多序列比对
│   ├── align_monthly.py              # MAFFT 逐文件多线程比对
│   └── qc_clean_msa.py               # 比对结果结构核对、错误 gap 列剔除、错位序列修复
├── 熵计算
│   ├── compute_entropy_monthly.py    # 逐位点香农熵 -> entropy_monthly/
│   └── smooth_entropy_monthly.py     # 滑动平均平滑 -> entropy_monthly_smoothed/
├── 表征模型
│   ├── owid_repr13.py                # 13 周窗 Transformer 自监督预训练（repr13）
│   ├── repr_model_selection.py       # 候选表征受控探针选型
│   ├── reprM_selection.py            # 周窗月均 vs 月度输入表征三方对比
│   └── gen_monthly_embeddings.py     # 生成各国月度表征 embeddings_monthly/
├── 熵预测
│   ├── train_entropy_forecast_monthly.py  # 月表征(+历史熵) -> 下一月平滑熵，含消融
│   ├── finalize_monthly_forecast.py       # 定稿 3 种子集成模型与指标导出
│   └── dump_val_predictions.py            # 验证集逐月预测导出
├── 事后解释
│   ├── policy_response_models.py     # 政策→病例/死亡 辅助响应模型（反事实联动用）
│   ├── factor_sensitivity.py         # 单因素控制变量比例扰动敏感性分析
│   ├── diagnose_speed_cap.py         # 接种速度上限处理方式的通道分解诊断
│   └── policy_combination_search.py  # 覆盖率×速度×管控 三因素组合网格寻优
├── data（见"数据与产物"）
└── README.md
```

## 环境依赖

- Python 3.10（本研究在 Anaconda 环境运行）
- Python 包见 `requirements.txt`：

```
pip install -r requirements.txt
```

- 多序列比对依赖外部工具 **MAFFT（Windows 版）**：下载 mafft-win 并解压，脚本中默认路径为
  `H:\Bioinfo\mafft-win\mafft.bat`，请按本机实际安装路径修改
  `align_monthly.py` 与 `qc_clean_msa.py` 顶部的 `MAFFT` 常量。

## 数据获取与准备

### 1. GISAID spike 核苷酸序列（未随仓库发布）

本研究原始序列来自 GISAID（spike 核苷酸序列合集，约 64.6GB）。**因数据体积远超
GitHub 限制，且 GISAID 使用条款不允许公开再分发，序列数据及其中间产物
（`GISAID_MSA_monthly/`、`MSA_monthly/`）均未上传。**

复现方式：注册 GISAID 账号（https://gisaid.org）并通过审核后，下载 spike
核苷酸序列合集（fasta），置于项目根目录并命名为 `spikenuc1116.fasta`
（或在 `extract_spike_monthly.py` 中修改输入路径）。

### 2. OWID 疫情指标（已随仓库发布）

- `owid_cleaned_weekly70_v3.xlsx`：清洗后的 OWID 周度面板（141 国 × 148 周，
  含疫苗覆盖率/接种速度、管控严格度、病例、死亡、传播率 Rt、人口结构与 GDP 等字段）。
  原始数据来源：Our World in Data COVID-19 dataset（https://ourworldindata.org/coronavirus，CC BY 4.0）。
- `owid_weekly_panel.pkl`：由上述 xlsx 整理得到的周频面板（表征模型与事后解释脚本直接读取）。

## 使用流程（按管线顺序）

以下命令均假设在项目根目录执行。

### 第一步：序列月度分组

```bash
python extract_spike_monthly.py chunk --i 0 --n 10   # 分 10 块并行处理，i=0..9
python extract_spike_monthly.py finalize             # 合并国名变体、汇总
python reorg_country_dirs.py                         # 目录归并整理
python dedupe_files.py                               # 全量去重复扫
```

产出 `GISAID_MSA_monthly/{国家}/{YYYY-MM}.fasta`（219 个国家/地区）。

### 第二步：多序列比对

```bash
python align_monthly.py    # MAFFT 多线程逐文件比对 -> MSA_monthly/
python qc_clean_msa.py     # 结构核对 + 错误 gap 列剔除 + 错位序列修复
```

### 第三步：月度香农熵

```bash
python compute_entropy_monthly.py    # 逐位点熵按月求和 -> entropy_monthly/
python smooth_entropy_monthly.py     # 5 点居中滑动平均 -> entropy_monthly_smoothed/
```

注：本仓库已附带这两个目录的熵结果（无序列内容），无序列数据时可直接使用。

### 第四步：OWID 表征模型

```bash
python owid_repr13.py data            # 构建窗口样本数据集
python owid_repr13.py train --cfg v1_ce   # 训练指定配置（可遍历配置）
python owid_repr13.py eval            # 汇总对比，选定最优 -> repr13_best.pt
python repr_model_selection.py        # 受控探针选型（国家识别 + 月熵 Ridge 探针）
python reprM_selection.py             # 周窗月均 vs 月度输入 三方对比
python gen_monthly_embeddings.py      # 生成月度表征 -> embeddings_monthly/
```

注：`embeddings_monthly/` 月度表征数据未随仓库发布，运行上述流程即可重新生成。

### 第五步：月度熵时间序列预测

```bash
python train_entropy_forecast_monthly.py --mode full --K 3     # 表征+历史熵
python train_entropy_forecast_monthly.py --mode ent_only       # 纯熵消融对照
python finalize_monthly_forecast.py --save_best                # 定稿 3 种子集成
python dump_val_predictions.py                                 # 导出验证集逐月预测
```

最优模型（K=3 残差架构、3 种子集成）验证集 R²≈0.822，表征贡献约 31%
（相比纯熵消融的 MSE 降幅占比）。

### 第六步：事后解释与组合寻优

```bash
python policy_response_models.py      # 训练 政策->病例/死亡 辅助响应模型（Ridge/HistGBR 选优）
python factor_sensitivity.py          # 单因素 ±10%…±100% 比例扰动敏感性分析
python diagnose_speed_cap.py          # 接种速度上限处理方式诊断（存档证据）
python policy_combination_search.py   # 覆盖率×速度×管控 组合网格寻优
```

事后解释的现实约束：接种速度扰动下每日覆盖率净增值 ≤ 1 剂/百人；
管控严格度值域 [0,100]，且观测值 <15 的国家-周不允许下调；病例/死亡经
辅助响应模型差分法联动反事实。

## 随仓库发布的模型权重与结果文件

| 文件 | 说明 |
| --- | --- |
| `repr13_best.pt` | 最优 OWID 表征模型（repr13 配置）权重 |
| `entropy_forecast_monthly_best.pt` | 月度熵预测最终模型（3 种子残差集成） |
| `entropy_forecast_monthly_entonly_best.pt` | 纯熵消融集成对比模型 |
| `policy_response_models.pkl` | 政策→病例/死亡辅助响应模型 |
| `entropy_monthly/`、`entropy_monthly_smoothed/` | 各国月度熵与平滑熵（`.npy`，dict: "YYYY-MM" -> float） |
| `entropy_forecast_monthly_metrics.json`、`entropy_forecast_monthly_per_country.csv` | 熵预测验证集指标 |
| `factor_effects_monthly.json/.csv`、`factor_effects_vaccine_era.json` | 单因素扰动效应 |
| `policy_combination_results.json`、`policy_recommendations.csv` | 组合寻优结果与最优策略 |
| `speed_cap_diagnostic.json` | 接种速度上限诊断结果 |
| `repr_model_selection.json`、`reprM_selection.json` | 表征选型对比结果 |
| `val_pred_monthly.npz`、`policy_response_val_pred.npz` | 验证集逐样本预测 |
| `aux_model_names.json` | 辅助模型清单 |

## 数据使用声明

- GISAID 序列数据的使用须遵守 GISAID 数据访问协议（https://gisaid.org/terms-of-use/），
  请勿公开再分发序列；本仓库不含任何序列数据。
- OWID 数据遵循 CC BY 4.0（https://ourworldindata.org/coronavirus）。
- 本仓库代码仅供学术研究使用。
