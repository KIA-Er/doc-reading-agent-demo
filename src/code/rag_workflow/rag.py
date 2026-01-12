import os
import sys
from typing import List, Dict, Any, Optional

current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
sys.path.append(project_root)

from loguru import logger
from src.code.data_base.database import VectorDatabase, vector_db
from src.code.embedding.embedding_model import JinaEmbeddingClient
from src.code.rerank.reranker import Reranker
from src.settings import settings
from src.code.visual_reasoner.model import VisionLanguageModel
import asyncio

from src.settings import settings

logger =logger.bind(module ="rag_workflow")


VECTOR_DATABASE_URI = "http://192.168.3.112:19530"
VECTOR_DATABASE_NAME = "default"
COLLECTION_NAME = "WENKAI_reading_agent_demo"

RERANKER_BASE_URL = "http://192.168.3.112:9907/v1/rerank"
RERANKER_MODEL_NAME = "jina/jina-rerank-m0"

class Retriever():
    def __init__(self):
        self.embedding_model = JinaEmbeddingClient()
        self.reranker = Reranker(
            baseurl=RERANKER_BASE_URL,
            model_name=RERANKER_MODEL_NAME,
            top_k=5,
        )
        self.vector_db = VectorDatabase(
            uri=VECTOR_DATABASE_URI,
            db_name=VECTOR_DATABASE_NAME,
            embedding_func=self.embedding_model.get_embedding,
        )
        self.vlm_model = VisionLanguageModel(
            model_name=settings.VLM_MODEL_NAME,
            url=settings.VLM_BASE_URL,
        )
        logger.info(f"RAG Retriever已就绪")

    async def retieve(self, query: str) -> str:
        # 嵌入查询并对文件进行向量检索
        related_results = await self.vector_db.query(
            query=query, 
            top_k=10)

        # 对检索结果进行重排序
        reranked_results = await self.reranker.rerank(
            query=query, 
            img_urls=[item['image_url'] for item in related_results[0]])
        
        #获取重排序后的结果URL列表
        result_urls = [ item['document']['text'] for item in reranked_results['results'] ]
        
        #传入VLM模型进行推理
        response = self.vlm_model.run(
            query=query,
            image_urls=result_urls,
        )
        return response

logger.disable("src.code.embedding")
logger.disable("src.code.visual_reasoner")
logger.disable("src.code.rerank")

retriever = Retriever()



while True:
    query = input("请输入您的问题：")
    if query == "exit":
        break
    response = asyncio.run(retriever.retieve(query=query))
    print(response)
