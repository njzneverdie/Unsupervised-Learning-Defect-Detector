"""凍結的 WRN50 backbone + forward hook,側錄 layer2 / layer3 中間層特徵。"""
from typing import Dict
import torch
import torch.nn as nn
import torchvision

from config import Config


class FeatureExtractor:
    """載入 ImageNet 預訓練 WideResNet-50,用 forward hook 抓 layer2、layer3 輸出。

    backbone 全程凍結(不訓練):它只是把影像翻譯成好特徵的固定翻譯機。
    """

    def __init__(self, cfg: Config, device):
        self.cfg = cfg
        self.device = device

        # 相容新舊 torchvision 的權重載入 API
        try:
            weights = torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
            model = torchvision.models.wide_resnet50_2(weights=weights)
        except AttributeError:
            model = torchvision.models.wide_resnet50_2(pretrained=True)

        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model

        # forward hook:在 layer2、layer3 算完的當下側錄輸出,不改動模型
        self._feats: Dict[str, torch.Tensor] = {}
        model.layer2.register_forward_hook(self._make_hook("layer2"))
        model.layer3.register_forward_hook(self._make_hook("layer3"))

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._feats[name] = out
        return hook

    @torch.no_grad()
    def extract(self, batch: torch.Tensor):
        """batch: (B, 3, t, t) → (layer2 特徵, layer3 特徵)。

        layer2: (B, 512, t/8,  t/8)
        layer3: (B, 1024, t/16, t/16)
        """
        self._feats.clear()
        _ = self.model(batch)                 # 前向到底,但我們只要 hook 側錄的中間層
        return self._feats["layer2"], self._feats["layer3"]
