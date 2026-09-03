# -*- coding: utf-8 -*-
"""compose_fig8.py — 以调整后的子图(figS14..figS19 独立PNG)重排版 fig8 组合图

figure-composer 流程: 子图已按 figure-style 逐张调整好(standalone PNG),
本脚本只做 outline -> compose_figure 拼接(盖字母) -> compose_crops 切割目检。
槽位高度按子图实际像素宽高比设定, 重采样失真 ≤±4%。
输出: figure_monthly_revised/fig8_data_overview.png (覆盖旧组合图)
"""
import json
import os
import sys

sys.path.insert(0, r"C:/Users/DFD/.agents/skills/figure-composer")
_kernel = {}
exec(open(r"C:/Users/DFD/.agents/skills/figure-composer/kernel.py",
          encoding="utf-8").read(), _kernel)

SUP = "figure_monthly_revised/supplementary"
OUT = "figure_monthly_revised/fig8_data_overview.png"
CROP_DIR = "_fig8_crops"
os.makedirs(CROP_DIR, exist_ok=True)

outline = {
    "claim": "数据覆盖219国、144万条去重后序列，经清洗筛选形成124国3262样本的建模数据集",
    "width_mm": 180,
    "ncol": 12,
    "row_heights_mm": [44.1, 45.8],
    "panels": [
        {"letter": "a", "role": "primary", "row": 0, "col": 0, "colspan": 4,
         "chart_family": "area timeline",
         "message": "月度序列量于2021年末达峰",
         "data_path": "seqcounts_monthly.csv",
         "ask": "全球月度去重后序列数面积图, 标注峰值与训练/验证分界"},
        {"letter": "b", "role": "primary", "row": 0, "col": 4, "colspan": 4,
         "chart_family": "log-log rank-size scatter",
         "message": "国家序列量呈长尾分布",
         "data_path": "seqcounts_monthly.csv",
         "ask": "rank-size 对数散点, 前15国76.7%份额带"},
        {"letter": "c", "role": "primary", "row": 0, "col": 8, "colspan": 4,
         "chart_family": "horizontal bar funnel",
         "message": "219国序列经筛选形成124国建模集",
         "data_path": "entropy_forecast_monthly_v2_metrics.json",
         "ask": "数据集筛选横向条图漏斗"},
        {"letter": "d", "role": "evidence", "row": 1, "col": 0, "colspan": 4,
         "chart_family": "heatmap + marginal line",
         "message": "国家×月熵观测覆盖与逐月观测国家比例",
         "data_path": "entropy_monthly/",
         "ask": "覆盖热图+边际曲线增强版"},
        {"letter": "e", "role": "evidence", "row": 1, "col": 4, "colspan": 4,
         "chart_family": "ECDF",
         "message": "七成国家熵观测达15月门槛",
         "data_path": "entropy_monthly/",
         "ask": "观测月数ECDF, ≥15月门槛阴影区"},
        {"letter": "f", "role": "evidence", "row": 1, "col": 8, "colspan": 4,
         "chart_family": "violin by year",
         "message": "2020年国家-月序列数偏低",
         "data_path": "seqcounts_monthly.csv",
         "ask": "分年份国家-月序列数小提琴图"},
    ],
}

panel_paths = {
    "a": f"{SUP}/figS14_data_timeline.png",
    "b": f"{SUP}/figS15_data_rank_size.png",
    "c": f"{SUP}/figS16_data_funnel.png",
    "d": f"{SUP}/figS17_entropy_coverage_heatmap.png",
    "e": f"{SUP}/figS18_entropy_observed_ecdf.png",
    "f": f"{SUP}/figS19_seqcount_violin.png",
}

out_path, (W, H) = _kernel["compose_figure"](outline, panel_paths, OUT)
print("composed:", out_path, W, "x", H)

from PIL import Image
comp = Image.open(OUT)
for L, box in _kernel["compose_crops"](outline).items():
    comp.crop(box).save(os.path.join(CROP_DIR, f"crop_{L}.png"))
print("crops saved to", CROP_DIR)
