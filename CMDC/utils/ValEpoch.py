from tqdm import tqdm
import torch
from torchnet.meter import AverageValueMeter
from utils.metrics import CMMeter
from utils.func import get_mem, list2device
import numpy as np
import os
from PIL import Image


def torch_save_gray_batch(batch_tensor, save_dir, prefix="gray_", ext="png"):
    """
    批量保存单通道黑白图（PyTorch Tensor）
    
    参数:
        batch_tensor: 输入批量Tensor，形状为 (B, 1, H, W)
        save_dir: 保存目录（自动创建）
        prefix: 文件名前缀（如"gray_"）
        ext: 图片格式（如"png"、"jpg"）
    """
    # 校验输入格式（必须是单通道批量图）
    assert batch_tensor.dim() == 4 and batch_tensor.shape[1] == 1, \
        f"输入需为单通道批量Tensor (B, 1, H, W)，当前形状: {batch_tensor.shape}"
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    batch_size = batch_tensor.shape[0]
    
    for i in range(batch_size):
        # 提取单张图并移除通道维度 (1, H, W) -> (H, W)
        tensor = batch_tensor[i].squeeze(0)  # 形状变为 (H, W)
        
        # 转换为numpy数组
        img_np = tensor.cpu().detach().numpy()
        
        # 归一化到0-255（支持float(0-1)或int(0-255)输入）
        if img_np.dtype in (np.float32, np.float64):
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.clip(0, 255).astype(np.uint8)
        else:
            img_np = img_np.clip(0, 255).astype(np.uint8)
        
        # 保存为单通道灰度图
        save_path = f"{save_dir}/{prefix}{i}.{ext}"
        Image.fromarray(img_np, mode="L").save(save_path)  # mode="L"表示单通道灰度图



class ValEpoch:
    def __init__(self, num_classes, net,
                criterion1,
                metric, device="cuda"):

        self.num_classes = num_classes
        self.net = net
        self.criterion = criterion1
        # self.criterion_consistency = criterion_consistency
        self.metric = metric
        self.device = device

        self._to_device()

    def _to_device(self):
        self.net.to(self.device)
        self.criterion.to(self.device)
        self.metric.to(self.device)

    @torch.no_grad()
    def run(self, dataloader):
        print(('\n' + '%10s' * 8) % ("val", 'gpu', 'loss', 'precision', 'recall', 'f1', 'iou', 'OA'))

        # 测试模式
        self.net.eval()

        # loss和指标
        loss_meter = AverageValueMeter()
        cm_meter = CMMeter()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, sample in pbar:
        # for step, (sample, _) in pbar:
            # x = x.to(self.device)
            x = list2device(sample['image'], self.device)
            label = sample['labels'].to(self.device)
            Chg,a,b = self.net(x)
            
            out = torch.argmax(Chg, dim=1, keepdim=True)  # 取最大概率通道 (B, 1, H, W)
            out = (out > 0.5).float() * 255  # 二值化并缩放到0-255
            out = out.cpu()
    
    # 处理label（确保形状匹配，转为单通道）
    # 假设label形状是(B, H, W)，转为(B, 1, H, W)以适配保存函数
            label_processed = label.unsqueeze(1).cpu()  # 增加通道维度
            label_processed = (label_processed > 0.5).float() * 255  # 二值化（如果需要）
    
    # 生成统一前缀（确保对应样本名称相同）
            prefix = f"step{step}_sample"
    
    # 保存Chg结果（out）和label，使用相同前缀
            torch_save_gray_batch(
        out, 
        "/home/data3/liangyizhou/semiCD/Ctestcd/vis/conwhu", 
        prefix=prefix
    )
            torch_save_gray_batch(
        label_processed,  # 适配binary函数的(B, H, W)输入
        "/home/data3/liangyizhou/semiCD/Ctestcd/vis/L/conwhu", 
        prefix=prefix
    )

            loss = self.criterion(Chg, label)
            loss_labeled_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_labeled_value)
            metrics = self.metric(Chg, label)
            cm_meter.add(metrics)
            precision, recall, f1, iou, oa = cm_meter.get_metrics()

            pbar.set_description(('%10s' * 2 + '%10.4g' * 6) % ("val", get_mem(), loss_labeled_value,
                                                                precision, recall, f1, iou, oa))
        precision, recall, f1, iou, oa = cm_meter.get_metrics()
        logs = {
            'loss': loss_meter.mean,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'oa': oa
        }
        return logs