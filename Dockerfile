FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖（ultralytics 需要 libgl1）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制服务代码
COPY service/ .

# 模型文件（best.pt）体积较大且被 .gitignore 排除，
# 请将训练好的模型文件放到 service/model/best.pt，或挂载外部卷：
#   -v /host/path/best.pt:/app/model/best.pt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]