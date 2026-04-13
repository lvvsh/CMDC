import torch
from thop import profile
# 导入你的模型，因为你是基于CutMix-CD改的，直接导入你的 student_net 即可
import models 

def main():
    # 1. 实例化你的模型 (只保留推理部分)
    # 假设你的模型类别名叫 ResNet50_CD，请根据你的实际代码修改
    model = models.ResNet50_CD(num_classes=2, pretrained=False)
    model.eval() # 切换到推理模式

    # 2. 构造虚拟输入 (假设输入是 256x256 的双时相图像)
    # 根据你之前的代码，你的模型输入似乎是一个包含两张图的 list 或 tuple
    input_t1 = torch.randn(1, 3, 256, 256)
    input_t2 = torch.randn(1, 3, 256, 256)
    
    # 3. 计算 FLOPs 和 Params
    # 如果你的模型 forward 函数接收的是一个 list: model([input_t1, input_t2])
    try:
        flops, params = profile(model, inputs=([input_t1, input_t2], ))
    except:
        # 如果你的模型 forward 接收的是两个独立的 tensor: model(input_t1, input_t2)
        flops, params = profile(model, inputs=(input_t1, input_t2))

    # 4. 打印结果
    print("="*30)
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")
    print("="*30)

if __name__ == "__main__":
    main()