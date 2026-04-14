from ultralytics import YOLO
import os

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 设定数据集路径（相对路径）
data_path = os.path.join(PROJECT_ROOT, 'datasets')

# 项目名称
project_name = 'tianyu_plant_project'

# 实验名称
# exp_name = 'exp1_base_model'


exp_name = 'exp2_augmented'


def train_model():
    print("开始训练模型...")
    # 加载模型
    model = YOLO('yolov8n-cls.pt')  # 使用预训练的YOLOv8n分类模型

    # 开始训练
    model.train(
        data=data_path,
        epochs=50,
        imgsz=224,
        batch=16,
        workers=0,
        project=project_name,
        name=exp_name,
        exist_ok=True,
        device='cpu',
        # 数据增强参数
        degrees=25.0,  # 随机旋转 +/- 25度 (模拟拍摄角度歪了)
        scale=0.5,  # 随机缩放 (模拟远近拍摄)
        fliplr=0.5,  # 50%概率左右翻转
        flipud=0.0,  # 上下翻转 (叶子通常是朝上的，这个可以设低点或0)
        hsv_h=0.015,  # 色调微调 (模拟不同光照)
        hsv_s=0.7,  # 饱和度干扰
        hsv_v=0.4,  # 亮度干扰
        mosaic=1.0,  # 马赛克增强 (把4张图拼在一起训练，极其有效！)
        erasing=0.4,  # 随机擦除 (模拟叶子被遮挡)
    )
    print("训练完成。")
    metrics = model.val()
    print(f"Top-1 准确率: {metrics.top1:.4f}")
    print("正在导出为ONNX格式模型...")
    # 导出模型为ONNX格式
    success = model.export(format='onnx')
    if success:
        print("模型成功导出为ONNX格式。")
    else:
        print("模型导出失败。")


if __name__ == "__main__":
    train_model()
