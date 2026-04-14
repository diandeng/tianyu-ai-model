import os
import shutil
import random

# 原始数据集路径
raw_dataset_dir = r"D:\Project\TianyuAI\PlantVillage-Dataset\raw\color"

# 输出数据集路径
output_dir = r"D:\Project\TianyuAI\tianyu-ai-model\datasets"

# 目标类别列表
target_classes = [
    # 1. 苹果 (Apple)
    "Apple___Apple_scab",  # 苹果黑星病
    "Apple___Black_rot",  # 苹果黑腐病
    "Apple___Cedar_apple_rust",  # 苹果锈病
    "Apple___healthy",  # 苹果健康

    # 2. 玉米 (Corn)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",  # 玉米灰斑病
    "Corn_(maize)___Common_rust_",  # 玉米锈病
    "Corn_(maize)___Northern_Leaf_Blight",  # 玉米大斑病
    "Corn_(maize)___healthy",  # 玉米健康

    # 3. 葡萄 (Grape)
    "Grape___Black_rot",  # 葡萄黑腐病
    "Grape___Esca_(Black_Measles)",  # 葡萄褐斑病
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",  # 葡萄叶枯病
    "Grape___healthy",  # 葡萄健康

    # 4. 桃子 (Peach)
    "Peach___Bacterial_spot",  # 桃细菌性穿孔病
    "Peach___healthy",  # 桃健康

    # 5. 辣椒 (Pepper)
    "Pepper,_bell___Bacterial_spot",  # 辣椒细菌性斑点病
    "Pepper,_bell___healthy",  # 辣椒健康

    # 6. 马铃薯 (Potato)
    "Potato___Early_blight",  # 马铃薯早疫病
    "Potato___Late_blight",  # 马铃薯晚疫病
    "Potato___healthy",  # 马铃薯健康

    # 7. 番茄 (Tomato)
    "Tomato___Bacterial_spot",  # 番茄细菌性斑点病
    "Tomato___Early_blight",  # 番茄早疫病
    "Tomato___Late_blight",  # 番茄晚疫病
    "Tomato___Leaf_Mold",  # 番茄叶霉病
    "Tomato___Septoria_leaf_spot",  # 番茄斑枯病
    "Tomato___Spider_mites Two-spotted_spider_mite",  # 番茄红蜘蛛
    "Tomato___Target_Spot",  # 番茄靶斑病
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",  # 番茄黄化曲叶病毒
    "Tomato___Tomato_mosaic_virus",  # 番茄花叶病毒
    "Tomato___healthy"  # 番茄健康
]

# 划分比例
train_ratio = 0.8


def split_dataset():
    # 检测原始路径是否存在
    if not os.path.exists(raw_dataset_dir):
        print(f"错误: 原始数据集路径不存在: {raw_dataset_dir}")
        return

    # 清除并重建输出目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'train'))
    os.makedirs(os.path.join(output_dir, 'val'))

    print("开始数据集划分...")

    for class_name in target_classes:
        src_class_path = os.path.join(raw_dataset_dir, class_name)
        if not os.path.exists(src_class_path):
            print(f"警告: 类别路径不存在, 跳过: {src_class_path}")
            continue

        # 获取该类别下的所有图片
        images = [f for f in os.listdir(src_class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        # 计算划分点
        split_point = int(len(images) * train_ratio)
        train_imgs = images[:split_point]
        val_imgs = images[split_point:]

        # 建立目标类别目录
        os.makedirs(os.path.join(output_dir, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', class_name), exist_ok=True)

        # 复制文件
        print(f"正在处理 {class_name}: 训练集 {len(train_imgs)} 张, 验证集 {len(val_imgs)} 张")

        for img in train_imgs:
            shutil.copy(os.path.join(src_class_path, img), os.path.join(output_dir, 'train', class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_class_path, img), os.path.join(output_dir, 'val', class_name, img))
    print("数据集划分完成!")


if __name__ == "__main__":
    split_dataset()
