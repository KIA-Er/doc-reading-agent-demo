import os
import sys
from pathlib import Path
import base64
import json
from typing import List, Dict, Any
from openai import OpenAI
import httpx
from httpx import AsyncClient, Timeout
import numpy as np
from loguru import logger
from src.settings import settings
from PIL import Image
import asyncio
from pdf2image import convert_from_path
from io import BytesIO

logger = logger.bind(name="JinaEmbedding客户端")

def convert_to_jpeg(images: List[Image.Image]) -> List[Image.Image]:
    processed_images = []

    for img in images:
        # 步骤 A: 确保颜色模式是 RGB (防止 CMYK 导致保存 JPEG 失败)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        buffer.seek(0)
        
        new_img = Image.open(buffer)
        processed_images.append(new_img)
    
    return processed_images

class JinaEmbeddingClient():
    def __init__(self):
        self.base_url = settings.JINA_EMBEDDING_BASE_URL
        self.api_key = settings.JINA_EMBEDDING_MODEL_API_KEY
        self.embedding_dim = settings.JINA_EMBEDDING_MODEL_DIMS
        self.embedding_name = settings.JINA_EMBEDDING_MODEL_NAME
        
        # self.base_url = settings.QWEN3_EMBEDDING_MODEL_BASE_URL
        # self.api_key = settings.QWEN3_EMBEDDING_MODEL_API_KEY
        # self.embedding_dim = None
        # self.embedding_name = settings.QWEN3_EMBEDDING_MODEL_NAME
        self.timeout = Timeout(60.0, connect=10.0)

        self.headers = {
            "Content-type": "application/json",
            "User-Agent": "wenkai_test"
        }
        
        logger.info(f"通过HTTP请求访问JinaEmbedding服务: {self.embedding_name} at {self.base_url} 成功！")
    
    async def get_embedding(self, text: str = "", *, image: Image.Image=None, is_base64=True) -> List[float]:
        """
        [异步] 获取多模态向量
        
        Args:
            text: 提示词文本
            image: PIL Image 对象 (可选)
        Returns:
            List[float]: 嵌入向量
        """

        content_block: List[Dict[str, Any]] = []

        if text:
            content_block.append(
                {
                    "type": "text",
                    "text":text 
                },
            )

        if image:
            image_http_url = ""#TODO：日后再添加，测试miniserve的静态文件服务器功能
            images_base64 = self._convert_to_base64(image)
            content_block.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{images_base64}"if is_base64 else f"{image_http_url}"
                    }
                }
            )
        
        if not content_block:
            logger.error(f"未提供文本或图片内容，无法获取嵌入向量！")
            raise ValueError("必须提供text或image内容至少一项！")
        
        payload = {
            "model": self.embedding_name,
            "messages": [
                {
                    "role": "user",
                    "content": content_block
                }
            ]
        }
        logger.info(f"payload构造完毕，前50字符: {str(payload)}...")

        async with AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    url=self.base_url,
                    headers= self.headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    raise ValueError(f"HTTP Error {response.status_code}: {response.text}")

                result = response.json()
                if "data" in result and len(result["data"])>0:
                    return result["data"][0]["embedding"]
                
            except httpx.RequestError as e:
                logger.warning(f"请求JinaEmbedding服务器时出现异常：{e}")
                raise 

    def _convert_to_base64(self, image: Image.Image) -> str:
        logger.info(f"正在将 1 张图片转换为 Base64 编码, 以便发送到 Jina Embedding 服务...")
        images_base64 = []
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        buffer.seek(0)
        img_bytes = buffer.read()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        images_base64.append(img_base64)

        logger.info(f"图片转换为 Base64 编码完成！")
        return images_base64[0]

    def _cal_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """用numpy计算两个向量的余弦相似度"""
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        dot_product = np.dot(v1,v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        #防止除零
        if norm_v1 ==0 or norm_v2 ==0:
            return 0.0
        
        return dot_product / (norm_v1 * norm_v2)


        
if __name__ == "__main__":
    jinaclient = JinaEmbeddingClient()

    root_path = Path.cwd()
    print(root_path)
    file_path = os.path.join(root_path, "示例数据", "test.pdf")
    pdf_doc = convert_from_path(file_path, first_page=1, last_page=1)
    images = convert_to_jpeg(pdf_doc)

    for img in images:
        embedding = asyncio.run(jinaclient.get_embedding(image=img))
        print(f"Embedding前20个维度: {embedding[:20]}")
    while True:
        user_input = input("请输入您的问题（输入 'exit' 退出）：")
        if user_input.lower() == 'exit':
            break
        user_input_embedding = asyncio.run(jinaclient.get_embedding(text=user_input))
        similarity = jinaclient._cal_cosine_similarity(user_input_embedding, embedding)
        print("余弦相似度为:", similarity)
        print(f"与文档相似"if similarity >0.5 else "与文档不相似")
    