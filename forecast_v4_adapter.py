# -*- coding: utf-8 -*-
"""forecast_v4_adapter.py — 月熵预测集成对事后解释管线的适配层
最终集成权重为 members 结构 (跨 K 成员 × 多种子), 与早期单一 RES +
全局 y_mu/y_sd 接口不同。本模块提供与解释管线兼容的预测接口:
  - 解释性反事实只扰动"表征通道" (Xe), 熵史/掩码/差分保持观测值;
  - 统一使用**未校准**集成预测 (静态校准仅用于性能评估口径);
  - predict_val 接受 (N_val, K_max, 64) 的扰动月表征, 按成员 K 切片后
    逐成员前向、原始尺度均值集成。
用法:
    fc = V4Forecaster()
    yhat = fc.predict_val(Em_val)      # Em_val: (len(fc.val_idx), K_max, 64)
"""
import numpy as np
import torch

import train_entropy_forecast_v2 as T
from train_entropy_forecast_monthly import TRAIN_END


class V4Forecaster:
    def __init__(self, ckpt="entropy_forecast_monthly_v2_best.pt", device=None):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.members = []
        for mem in ck["members"]:
            D = T.prepare_data(mem["K"], mem["target_norm"], mem["delta"],
                               "full", ent_dir=mem.get("ent_dir"))
            nets = []
            for st in mem["ensemble_state_dicts"]:
                net = T.make_model(mem["model"], mem["K"], mem["hidden"],
                                   mem["layers"], mem["dropout"],
                                   mem["F"]).to(self.device)
                net.load_state_dict(st)
                net.eval()
                nets.append(net)
            self.members.append(dict(K=mem["K"], delta=bool(mem["delta"]),
                                     D=D, nets=nets))
        m0 = self.members[0]["D"]
        va = m0["ts"] > TRAIN_END
        self.val_idx = np.nonzero(va)[0]
        self.y = m0["y"]
        self.cs = m0["cs"]
        self.ts = m0["ts"]
        self.countries = m0["countries"]
        self.K_max = max(mb["K"] for mb in self.members)

    def predict_val(self, Em_val):
        """Em_val: (len(val_idx), K_max, 64) -> (len(val_idx),) 原始尺度预测"""
        outs = []
        vi = self.val_idx
        for mb in self.members:
            K, D = mb["K"], mb["D"]
            Xf = D["Xf"][vi].copy()
            Xf[..., :64] = Em_val[:, -K:, :].astype(np.float32)
            X = torch.from_numpy(Xf).to(self.device)
            with torch.no_grad():
                pz = torch.stack([net(X) for net in mb["nets"]]) \
                    .mean(0).cpu().numpy()
            outs.append(T.Norm.inverse(pz.astype(np.float64),
                                       D["mu"][vi], D["sd"][vi]))
        return np.clip(np.mean(np.stack(outs), axis=0), 0, None)

    def anchor_r2(self, Em_val_obs):
        """δ=0 锚点: 观测表征下的 pooled R² (未校准, 全量验证样本)"""
        y0 = self.predict_val(Em_val_obs)
        yv = self.y[self.val_idx]
        return float(1 - ((y0 - yv) ** 2).sum()
                     / ((yv - yv.mean()) ** 2).sum()), y0
