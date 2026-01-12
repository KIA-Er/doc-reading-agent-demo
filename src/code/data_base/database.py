import os
import sys
from typing import List, Optional, Any, Dict, Callable, Union
from pydantic import BaseModel, Field
from loguru import logger
from pathlib import Path
from pymilvus import MilvusClient, DataType
import numpy as np
import uuid
import asyncio
from src.settings import settings
from src.code.embedding.embedding_model import JinaEmbeddingClient, convert_from_path, convert_to_jpeg

logger = logger.bind(module="rag_database")

VECTOR_DATABASE_URI = "http://192.168.3.112:19530"
VECTOR_DATABASE_NAME = "default"
COLLECTION_NAME = "WENKAI_reading_agent_demo"

class VectorSchema(BaseModel):
    id: Optional[int] = Field(default=None, description="文档唯一标识符")
    vector: List[float]
    page_index: Optional[int]
    image_url: Optional[str]= Field(default=None, description="图片的URL")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="附加的元数据字段")

class VectorDatabase:
    def __init__(
            self,
            uri: str = VECTOR_DATABASE_URI,
            db_name: str = VECTOR_DATABASE_NAME,
            *,
            embedding_func: Callable[[str | List[str]], List[List[float]]] = None,
            vlm= None,
            collection_name: str = COLLECTION_NAME,
            vector_dim: int = settings.JINA_EMBEDDING_MODEL_DIMS,
            ):

        self.embedding_func = embedding_func
        self.client = MilvusClient(
            uri=uri,
            db_name=db_name,
        )
        self.vector_dim = vector_dim

        if self.has_collection(collection_name):
            self.client.load_collection(collection_name)

    def create_collection(self, collection_name: str):
        logger.info(f"KAIEr:创建Milvus集合: {collection_name}")
        
        if self.has_collection(collection_name):
            raise ValueError(f"collection{collection_name}已存在，无法创建。")
        
        schema = self.client.create_schema(
            auto_id = True,
            enable_dynamic_field = True,# 允许以后存点别的杂七杂八的 metadata
            description = "KIAEr文档定位Agent索引",
        )

        #核心字段
        schema.add_field(
            field_name="id",
            datatype = DataType.INT64,
            is_primary = True,
            description = "文档唯一标识符,由Milvus自动生成",
        )

        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim = self.vector_dim,
            description="文档向量",
        )


        #业务元数据（Scalar Fields）
        schema.add_field(
            field_name="page_index",
            datatype= DataType.INT64,
        )

        schema.add_field(
            field_name="image_url",
            datatype= DataType.VARCHAR,
            max_length=256,
        )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            properties={"allow_insert_auto_id": "true"},
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="COSINE",
            index_type="AUTOINDEX",
            index_name="vector_index"
        )

        self.client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        
        logger.info(f"KIAEr:collection {collection_name} created")

    def delete_collection(self, collection_name: str):
        logger.info(f"KIAEr:删除Milvus集合: {collection_name}")

        if not self.has_collection(collection_name):
            raise ValueError(f"collection {collection_name} 不存在，无法删除。")
        self.client.drop_collection(collection_name)

    def insert_vectors(self,collection_name: str, vectors: Union[VectorSchema, List[VectorSchema]], metadatas: List[Dict[str, Any]] = None):
        
        insert_count = self.client.insert(
            collection_name=collection_name,
            data=vectors,
        )
        logger.info(f"KIAEr:插入 {insert_count} 条向量数据到集合 {collection_name}")
        
        return insert_count

    async def query(self, query : str, top_k: int = 10):

        vector = await self.embedding_func(query)
        search_result = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[vector],
            limit=top_k,
            output_fields=["id", "vector", "page_index", "image_url"],
        )

        return search_result

    async def add_documents(self, file_path: str):
        """
        将pdf或者其他格式的文件先转换为jpeg并添加元数据得到List[VectorSchema]，后调用embedding_func转换为向量并存入数据库
        """
        
        pdf_doc = convert_from_path(file_path, first_page=1)
        images = convert_to_jpeg(pdf_doc)
        logger.info(f"成功获取{len(images)}张pdf，并将其转换为JEPG图片")

        vectors: List[VectorSchema]=[]
    
        for idx, img in enumerate(images):
            vectors.append(
                VectorSchema(
                    id=idx,
                    vector=await self.embedding_func(image=img),
                    page_index=idx + 1,
                    image_url=f"/mnt/ssd2/steins/wenkai/project/doc-reading-agent-demo/demo_data_images/test_{idx+1}.jpeg"
                )
            )

        processed_vectors: List[Dict[str,Any]] = []
        for vec in vectors:
            processed_vectors.append(vec.model_dump())
        
        insert_count = self.insert_vectors(collection_name=COLLECTION_NAME, vectors=processed_vectors)
        
        logger.info(f"KIAEr:已添加文件 {file_path} 到向量数据库，共 {insert_count} 页。")

        return insert_count
    
    def get_page_index_by_image_url(self, collection_name: str=COLLECTION_NAME, image_url: str=None) -> Optional[int]:
        """
        根据 image_url 精确查询对应的 page_index
        
        Args:
            collection_name: 集合名称
            image_url: 要查询的图片路径字符串
            
        Returns:
            int: 对应的页码，如果没找到返回 None
        """
        filter_expr = f'image_url == "{image_url}"'
        
        try: 
            res = self.client.query(
                collection_name=collection_name,
                filter=filter_expr,
                output_fields=["page_index"], # 只返回需要的字段
                limit=1 # 既然是精确查找，只要一条
            )
            
            if res and len(res) > 0:
                return res[0].get("page_index")
            else:
                logger.warning(f"未找到 image_url 为 {image_url} 的记录")
                return None
                
        except Exception as e:
            logger.error(f"查询 page_index 失败: {e}")
            return None
        
    def get_page_indexes_by_image_urls(self, collection_name: str=COLLECTION_NAME, image_urls: List[str]=[]) -> List[Optional[int]]:

       pass

    def has_collection(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name)


emb_model = JinaEmbeddingClient()
vector_db = VectorDatabase(
    uri=VECTOR_DATABASE_URI,
    db_name=VECTOR_DATABASE_NAME,
    embedding_func=emb_model.get_embedding,
    )

root_path = Path.cwd()
file_path = os.path.join(root_path, "demo_data", "test.pdf")

# vector_db.delete_collection(COLLECTION_NAME)
# vector_db.create_collection(COLLECTION_NAME)
# pdf_doc = convert_from_path(file_path, first_page=1, last_page=1)
# img = convert_to_jpeg(pdf_doc)[0]
if __name__ == "__main__":
    
    logger.disable("src.code.embedding")
    print("Milvus集合列表:", vector_db.client.list_collections())
    asyncio.run(vector_db.add_documents(file_path=file_path))

    # asyncio.run(vector_db.add_documents(file_path=file_path))

    # result = asyncio.run(vector_db.query(query="我想知道第二章  采购需求的内容。", top_k=10))
    # best_one_page =result[0][0]['page_index']

