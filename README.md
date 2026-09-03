# SARS-CoV-2_mutation_analysis_transformer

基于 Transformer 表征与时间序列预测的新冠病毒（SARS-CoV-2）变异速度影响因素分析管线。

本项目以 GISAID 的 spike 核苷酸序列为变异信息源、以 Our World in Data（OWID）疫情指标面板为因素信息源，依次完成：序列月度分组与多序列比对 → 逐位点香农信息熵计算（变异速度代理指标）→ OWID 多维指标 Transformer 自监督表征 → 月度熵时间序列预测 → 基于控制变量扰动的事后解释与"疫苗接种 × 政策管控"组合策略寻优。

## 功能模块

| 模块 | 功能 |
| --- | --- |
| 序列处理 | 从 64.6GB GISAID spike 序列合集筛选、按月归档、去重，构建 219 个国家/地区的月度序列集 |
| 多序列比对 | MAFFT 多线程逐月比对（含超大文件分块比对 + merge 合并策略）与比对质控清洗 |
| 熵计算 | 逐位点香农熵按月求和，5 点居中滑动平均平滑 |
| 表征模型 | 13 周滑动窗口 Transformer 自监督预训练（国别分类 + 监督对比联合辅助任务），受控探针选型，生成 64 维月度表征（无前视） |
| 熵预测 | 跨配置持续性残差集成（K=5 与 K=3 双成员、各 3 种子）+ 逐国仿射静态校准，以月度表征 + 历史熵预测下一月平滑熵（验证集 pooled R² 0.832 未校准 / 0.826 校准） |
| 事后解释 | 单因素比例扰动敏感性分析（现实约束 + 辅助模型联动）、三因素组合网格寻优与六国情景轨迹分析 |

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
│   ├── owid_repr13.py                # 13 周窗 Transformer 自监督预训练（最终采用配置）
│   ├── repr_model_selection.py       # 候选表征受控探针选型
│   ├── owid_reprM.py                 # 月度输入表征（选型对比实验）
│   ├── reprM_selection.py            # 周窗月均 vs 月度输入表征三方对比
│   ├── gen_monthly_embeddings.py     # 生成各国月度表征 embeddings_monthly/（最终采用）
│   └── gen_monthly_embeddings_v2.py  # 月度输入模型的表征生成（对比实验用）
├── 熵预测
│   ├── train_entropy_forecast_monthly.py  # 数据构建与模型组件（被下述脚本复用）
│   ├── search_forecast_v2.py              # 分阶段配置搜索（A/B/C）与集成评估
│   ├── train_entropy_forecast_v2.py       # 残差架构训练器（搜索与定稿共用）
│   ├── finalize_monthly_forecast_v2.py    # 定稿跨配置集成、逐国静态校准与指标导出
│   └── dump_val_predictions_v2.py         # 验证集逐月预测导出
├── 事后解释
│   ├── policy_response_models.py     # 政策→病例/死亡 辅助响应模型（反事实联动用）
│   ├── forecast_v4_adapter.py        # members 结构集成对解释管线的预测接口适配
│   ├── factor_sensitivity_v2.py      # 单因素控制变量比例扰动敏感性分析
│   ├── policy_combination_search_v2.py  # 覆盖率×速度×管控 三因素组合网格寻优
│   └── compute_country_scenarios_v2.py  # 代表国家最优组合情景轨迹计算
├── 图表绘制
│   ├── plot_explanation_figures_v2.py   # 事后解释主图（fig0–fig7）
│   ├── plot_data_overview.py            # 数据分布/质量/清洗子图（figS14–figS21）
│   ├── compose_fig8.py                  # 数据概览组合图（fig8，需本地排版辅助）
│   ├── plot_repr_effect.py              # 表征效果组合图（fig9、figS22–figS27）
│   ├── compute_repr_effect_extra.py     # 表征效果量化指标（repr_effect_extra.json）
│   ├── plot_supplementary_figures_v2.py # 模型性能/熵演变/组合分析子图（figS1、figS3–figS8）
│   ├── plot_figS2_revised.py            # 分国家验证 R² 分布（figS2）
│   ├── plot_architecture_v2.py          # 模型结构示意图（figS9、figS10）
│   ├── plot_country_panels_v2.py        # 六国轨迹/剂量-响应/杠杆面板（figS11–figS13）
│   └── plot_forecast_eval_extra.py      # 残差诊断与逐月误差（figS28、figS29）
├── data（见"数据与产物"）
└── README.md
```

注：部分文件名中的 `_v2` 等后缀为管线演进过程中的历史命名，对应报告所述的最终模型与最终结果。

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
python reprM_selection.py             # 周窗月均 vs 月度输入 三方对比（对比实验）
python gen_monthly_embeddings.py      # 生成月度表征 -> embeddings_monthly/
```

注：`embeddings_monthly/` 月度表征数据未随仓库发布，运行上述流程即可重新生成。

### 第五步：月度熵时间序列预测

```bash
python search_forecast_v2.py            # 分阶段配置搜索（A 粗筛 -> B 细化 -> C 集成评估）
python finalize_monthly_forecast_v2.py  # 定稿跨配置集成 + 逐国静态校准 + 指标导出
python dump_val_predictions_v2.py       # 导出验证集逐月预测
```

最终模型为持续性残差网络的跨配置集成：K=5（隐藏层 256）与 K=3（隐藏层 128）双成员，
各 3 个随机种子在标准化空间取均值，预测后施加逐国仿射静态校准（岭收缩 λ=1）。
验证集（2022-08 至 2022-12，106 国评估口径，n=477）pooled R² 为 0.8320（未校准）/
0.8257（静态校准），纯熵消融集成 R² 为 0.7466，表征贡献 33.7%（验证集 MSE 相对下降率），
朴素持续性基线 R² 为 0.8311；校准后分国家 R²≥0.8 的国家由 17 个增至 25 个。

### 第六步：事后解释与组合寻优

```bash
python policy_response_models.py         # 训练 政策->病例/死亡 辅助响应模型（Ridge/HistGBR 选优）
python factor_sensitivity_v2.py          # 单因素 ±10%…±100% 比例扰动敏感性分析
python policy_combination_search_v2.py   # 覆盖率×速度×管控 组合网格寻优
python compute_country_scenarios_v2.py   # 代表国家最优组合情景轨迹
```

事后解释的现实约束：接种速度扰动下每日覆盖率净增值 ≤ 1 剂/百人（速度列余量与联动
覆盖率轨迹余量双重钳制，速度变化按 7 倍累计联动覆盖率轨迹）；管控严格度增量钳制在
值域 [0,100] 内，且观测值 <15 的国家-周不允许下调；病例/死亡经辅助响应模型差分法
联动反事实；检测量因素按研究设计剔除，人口结构、GDP 等静态背景不扰动。
组合寻优网格：覆盖率 ×{0.5,0.75,1,1.5,2,3}、接种速度 ×{0.5,1,1.5,2,3,5}、
管控 ×{0.5,0.75,1,1.25,1.5,2}；速度高档按日净增约束逐国判定可达性并降档，
精确平局时优先总变化最小的组合。全局最优组合为覆盖率 ×3、接种速度 ×2、管控 ×0.5
（覆盖 11.3% 验证样本，平均预测熵降幅 6.07%、中位 5.39%）。

### 第七步：图表绘制（可选）

```bash
python plot_explanation_figures_v2.py    # 事后解释主图 fig0–fig7
python plot_data_overview.py             # 数据概览子图 figS14–figS21
python compose_fig8.py                   # fig8 组合图（依赖本地 figure-composer 排版辅助，路径见脚本头部）
python plot_repr_effect.py               # fig9 与 figS22–figS27（先运行 compute_repr_effect_extra.py）
python plot_supplementary_figures_v2.py  # figS1、figS3–figS8
python plot_figS2_revised.py             # figS2
python plot_architecture_v2.py           # figS9、figS10
python plot_country_panels_v2.py         # figS11–figS13
python plot_forecast_eval_extra.py       # figS28、figS29
```

图表输出至 `figure_monthly_revised/`（按仓库 .gitignore 设置，图片文件不随仓库发布，
运行上述脚本即可重新生成）。

## 随仓库发布的模型权重与结果文件

| 文件 | 说明 |
| --- | --- |
| `repr13_best.pt` | 最优 OWID 表征模型（13 周窗配置）权重 |
| `reprM_best.pt` | 月度输入表征模型权重（选型对比实验） |
| `entropy_forecast_monthly_v2_best.pt` | 月度熵预测最终模型（跨配置残差集成 + 逐国静态校准） |
| `entropy_forecast_monthly_v2_entonly_best.pt` | 纯熵消融集成对比模型 |
| `policy_response_models.pkl` | 政策→病例/死亡辅助响应模型 |
| `entropy_monthly/`、`entropy_monthly_smoothed/` | 各国月度熵与平滑熵（`.npy`，dict: "YYYY-MM" -> float） |
| `entropy_forecast_monthly_v2_metrics.json`、`entropy_forecast_monthly_v2_per_country.csv` | 熵预测验证集指标（全量与分国家） |
| `forecast_v2_best_config.json`、`forecast_v2_stageA_top5.json` | 预测模型配置搜索记录 |
| `factor_effects_monthly_v2.json/.csv`、`factor_effects_vaccine_era_v2.json` | 单因素扰动效应 |
| `policy_combination_results_v2.json`、`policy_recommendations_v2.csv` | 组合寻优结果与最优策略 |
| `country_scenarios_v2.npz`、`country_scenarios_v2_meta.json` | 代表国家最优组合情景轨迹 |
| `seqcounts_monthly.csv` | 各国逐月去重后序列量统计 |
| `repr_model_selection.json`、`reprM_selection.json`、`repr_effect_extra.json` | 表征选型与效果量化结果 |
| `val_pred_monthly_v2.npz`、`policy_response_val_pred.npz` | 验证集逐样本预测 |
| `aux_model_names.json` | 辅助模型清单 |

## 数据使用声明

- GISAID 序列数据的使用须遵守 GISAID 数据访问协议（https://gisaid.org/terms-of-use/），
  请勿公开再分发序列；本仓库不含任何序列数据。
- OWID 数据遵循 CC BY 4.0（https://ourworldindata.org/coronavirus）。
- 本仓库代码仅供学术研究使用。
