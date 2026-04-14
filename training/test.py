from ultralytics import YOLO
import os

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 项目根目录
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 项目名称
project_name = 'tianyu_plant_project'

# 实验名称
exp_name = 'exp1_base_model'

# 模型路径（相对路径）
model_path = os.path.join(SCRIPT_DIR, 'runs', 'classify', project_name, exp_name, 'weights', 'best.pt')

# 测试图像路径（相对路径）
test_image_path = os.path.join(PROJECT_ROOT, 'datasets', 'train', 'Tomato___Bacterial_spot', '0a6d40e4-75d6-4659-8bc1-22f47cdb2ca8___GCREC_Bact.Sp 6247.JPG')


def test_model():
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先训练模型或检查路径是否正确。")
        return
    if not os.path.exists(test_image_path):
        print(f"错误: 找不到测试图像 {test_image_path}")
        return
    # 加载模型
    model = YOLO(model_path)

    print(f"正在预测图片: {test_image_path}")

    # 开始预测
    results = model(test_image_path)

    # 解析结果
    for result in results:
        # 获取概率最高的类别索引
        top1_index = result.probs.top1
        # 获取置信度
        confidence = result.probs.top1conf.item()
        # 获取类别名称
        class_name = result.names[top1_index]

        print("-" * 30)
        print(f"预测结果: 【{class_name}】")
        print(f"置信度:   {confidence:.2%}")
        print("-" * 30)

        # 置信度太低
        if confidence < 0.5:
            print("置信度较低，模型可能不确定。")


if __name__ == '__main__':
    test_model()
