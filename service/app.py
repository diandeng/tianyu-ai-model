import os
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException
# AI 模型相关库
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO
# 爬虫相关库
from bs4 import BeautifulSoup
import html2text
from urllib.parse import urljoin
import json

# 模型目录
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "model", "best.pt")

# 初始化FastAPI应用
app = FastAPI(title="Tianyu AI Disease Detection Service")

# 加载YOLO模型
print("正在加载模型...")
if os.path.exists(model_path):
    model = YOLO(model_path)
    print("模型加载成功！")
else:
    print("模型文件未找到，请确保模型文件存在于 'model/best.pt' 路径下。")
    model = None


# 定义输入数据模型
class ImageRequest(BaseModel):
    url: str  # 图片URL


@app.get("/")
def home():
    return {"msg": "Tianyu AI Model Service is Running!"}


@app.post("/predict")
def predict(req: ImageRequest):
    """
    进行图像疾病检测预测
    接收图片URL -> 下载图片 -> 进行预测 -> 返回结果
    :param req: ImageRequest
    :return: 预测结果
    """
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    # 下载图片
    try:
        print(f"正在下载图片: {req.url}")
        response = requests.get(req.url, timeout=10)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="无法下载图片，请检查 URL")
        # 转换为PIL图像对象
        image = Image.open(BytesIO(response.content))
        # 进行预测
        results = model.predict(image, save=False, verbose=False)

        # 解析结果
        result = results[0]
        top1_index = result.probs.top1
        top1_conf = result.probs.top1conf.item()
        class_name = result.names[top1_index]
        print(f"预测完成: {class_name} (置信度: {top1_conf:.4f})")

        # 返回 JSON
        return {
            "code": 200,
            "msg": "success",
            "data": {
                "class": class_name,  # 病害名称 (英文)
                "confidence": top1_conf,  # 置信度 (0.0 - 1.0)
                "is_confident": top1_conf > 0.8  # 是否自信 (给 Java 做判断用)
            }
        }
    except Exception as e:
        print(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/crawl/news")
def crawl_farmer_news(page: int = 0, limit: int = 10):
    """
    抓取农民日报-三农头条资讯
    :param page: 页码, 从0开始 (默认 0)
    :param limit: 每页数量 (默认 10)
    :return: 新闻列表
    """
    base_list_url = f"https://www.farmer.com.cn/farmer/xw/sntt/NewsList_{page}.json"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        # 1. 获取列表页 JSON
        response = requests.get(base_list_url, headers=headers, timeout=10)
        response.raise_for_status()

        # 网站返回的可能是 JSONP 格式或者带 BOM，安全起见直接解析 JSON
        data = response.json()
        info_list = data.get("info", [])

        results = []
        count = 0

        # 2. 初始化 Markdown 转换器
        h2t = html2text.HTML2Text()
        h2t.ignore_links = False
        h2t.ignore_images = False
        h2t.body_width = 0  # 不限制换行

        # 3. 遍历列表，获取正文
        for item in info_list:
            if count >= limit:
                break

            source_id = item.get("id")
            # 使用 ovtitle 获取无 html 标签的纯净标题
            title = item.get("ovtitle") or item.get("title")
            description = item.get("description", "")

            # 处理标签 (转成 List)
            raw_tags = item.get("Tags", "")
            tags = [tag.strip() for tag in raw_tags.split(',')] if raw_tags else []

            # 处理相对路径 URL
            raw_url = item.get("url", "")
            article_url = urljoin(base_list_url, raw_url)

            raw_thumb = item.get("thumb_image", "")
            cover_img = urljoin(base_list_url, raw_thumb) if raw_thumb else ""

            create_time = item.get("createTime", "")
            issue_time = item.get("issueTime", "")

            author = item.get("author", "")

            # --- 开始抓取正文 ---
            content_md = ""
            if raw_url:
                try:
                    art_resp = requests.get(article_url, headers=headers, timeout=10)
                    art_resp.encoding = 'utf-8'  # 农民日报一般是 utf-8
                    soup = BeautifulSoup(art_resp.text, 'html.parser')

                    # 找到目标 class="textList"
                    text_div = soup.find('div', class_='textList')
                    if text_div:
                        # 剔除可能存在的原网页分享按钮、广告等杂乱元素（可根据实际情况增加）
                        for bad_tag in text_div.find_all(['script', 'style']):
                            bad_tag.decompose()

                        # 告诉 h2t 基础 URL，这样文章里的相对图片路径就能转成绝对路径
                        h2t.baseurl = article_url
                        # 转换为 Markdown
                        content_md = h2t.handle(str(text_div))
                except Exception as e:
                    print(f"抓取正文失败 [{article_url}]: {e}")
                    content_md = "正文抓取失败"

            # 4. 组装单条数据
            news_dict = {
                "source_id": str(source_id),
                "title": title,
                "description": description,
                "cover_img": cover_img,
                "content_md": content_md,
                "tags": tags,  # List 格式，Java 接收后可转 JSON 存入
                "source_url": article_url,
                "create_time": create_time,
                "issue_time": issue_time,
                "author": author
            }
            results.append(news_dict)
            count += 1

        return {
            "code": 200,
            "msg": "success",
            "data": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    # 调试模式启动
    import uvicorn

    # 监听 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8000)
