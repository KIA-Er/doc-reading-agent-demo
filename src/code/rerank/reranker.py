from src.settings import settings
from loguru import logger
from httpx import AsyncClient, RequestError, Timeout
from typing import List, Dict, Any
import src.code.embedding
from PIL import Image
from io import BytesIO
import asyncio

logger = logger.bind(module="JINA_Reranker")

RERANKER_BASE_URL = "http://192.168.3.112:9907/v1/rerank"
RERANKER_MODEL_NAME = "jina/jina-rerank-m0"

class Reranker():
    def __init__(
            self, 
            baseurl: str = RERANKER_BASE_URL, 
            api_key: str = "", 
            model_name: str = RERANKER_MODEL_NAME,
            return_documents: bool = False,
            top_k: int =10,
            ):
        self.base_url = baseurl if baseurl else settings.JINA_RERANKER_MODEL_BASE_URL
        self.api_key = api_key if api_key else None
        self.reranker_name = model_name
        self.return_documents = "True" if return_documents  else "False"
        self.top_k = top_k if top_k else 5

        self.timeout = Timeout(60.0, connect=10.0)
        self.headers = {
            "Content-type": "application/json",
            "User-Agent": "wenkai_test"
        }

    async def rerank(self, query: str = "", *, img_urls: List[str] = None) -> list:#TODO：日后再添加，测试miniserve的静态文件服务器功能
        
        if not img_urls:
            logger.error(f"未提供图片内容，无法进行重排序！")
            raise ValueError("必须提供待重排序的image列表！")
        
        payload = {
            "model": self.reranker_name,
            "query": query,
            "documents": img_urls, # 传入URL地址
            # "documents": [
            #     "https://jina.ai/blog-banner/using-deepseek-r1-reasoning-model-in-deepsearch.webp"
            #     ],
            "top_n": self.top_k,
            "return_documents": f"{self.return_documents}",
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
                return result
                
            except RequestError as e:
                logger.warning(f"请求JinaEmbedding服务器时出现异常：{e}")
                raise 

# 测试代码
reranker = Reranker(return_documents=True)

if __name__ == "__main__":
    response = asyncio.run(reranker.rerank(
        query="""我想了解一下采购需求有什么内容？""", 
        img_urls=[f"/mnt/ssd2/steins/wenkai/project/doc-reading-agent-demo/demo_data_images/test_{i}.jpeg"for i in range(1,67)]
        ))
    indexes = [ item['index'] for item in response['results']]
    print(response['results'])