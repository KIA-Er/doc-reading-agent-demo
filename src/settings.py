"""
项目配置模块
加载 .env 环境变量并提供统一的配置访问
"""
from pathlib import Path
from dotenv import load_dotenv
import os


# 加载 .env 文件
def load_env():
    """加载根目录下的 .env 文件"""
    # 项目根目录
    ROOT_DIR = Path(__file__).resolve().parent.parent
    ENV_FILE = ROOT_DIR / ".env"
    
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)
        return True
    else:
        return False


# 执行环境变量加载
load_env()

# 环境变量访问接口
class Settings:
    """配置类，封装所有环境变量访问"""
    
    # 大语言模型配置
    @property
    def LLM_BASE_URL(self) -> str:
        return os.getenv("LLM_BASE_URL", "http://localhost:9902/v1")
    
    @property
    def LLM_NAME(self) -> str:
        return os.getenv("LLM_NAME", "Qwen/Qwen3-30B-A3B")
    
    @property
    def LLM_API_KEY(self) -> str:
        return os.getenv("LLM_API_KEY", "EMPTY")
    
    @property
    def OPENAI_API_KEY(self) -> str:
        return os.getenv("OPENAI_API_KEY", "EMPTY")
    
    @property
    def OPENAI_API_BASE_URL(self) -> str:
        return os.getenv("OPENAI_API_BASE_URL", "EMPTY")
    
    @property
    def DEFAULT_MODEL_TYPE(self) -> str:
        return os.getenv("DEFAULT_MODEL_TYPE", "EMPTY")
    
    # 多模态模型配置
    @property
    def VLM_BASE_URL(self) -> str:
        return os.getenv("VLM_BASE_URL", "http://localhost:9904/v1")
    
    @property
    def VLM_MODEL_NAME(self) -> str:
        return os.getenv("VLM_MODEL_NAME", "OpenBMB/MiniCPM-V-4")
    
    @property
    def VLM_API_KEY(self) -> str:
        return os.getenv("VLM_API_KEY", "EMPTY")
    
    # Qwen3 Embedding 模型配置
    @property
    def QWEN3_EMBEDDING_MODEL_BASE_URL(self) -> str:
        return os.getenv("QWEN3_EMBEDDING_MODEL_BASE_URL", "http://localhost:9901/v1")
    
    @property
    def QWEN3_EMBEDDING_MODEL_NAME(self) -> str:
        return os.getenv("QWEN3_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
    
    @property
    def QWEN3_EMBEDDING_MODEL_API_KEY(self) -> str:
        return os.getenv("QWEN3_EMBEDDING_MODEL_API_KEY", "EMPTY")
    
    @property
    def QWEN3_EMBEDDING_MODEL_DIMS(self) -> int:
        return int(os.getenv("QWEN3_EMBEDDING_MODEL_DIMS", "2560"))
    
    # Jina Embedding 模型配置
    @property
    def JINA_EMBEDDING_BASE_URL(self) -> str:
        return os.getenv("JINA_EMBEDDING_BASE_URL", "http://localhost:9908/v1/embeddings")
    
    @property
    def JINA_EMBEDDING_MODEL_NAME(self) -> str:
        return os.getenv("JINA_EMBEDDING_MODEL_NAME", "jinaai/jina-embeddings-v4-vllm-retrieval")
    
    @property
    def JINA_EMBEDDING_MODEL_API_KEY(self) -> str:
        return os.getenv("JINA_EMBEDDING_MODEL_API_KEY", "EMPTY")
    
    @property
    def JINA_EMBEDDING_MODEL_DIMS(self) -> int:
        return int(os.getenv("JINA_EMBEDDING_MODEL_DIMS", "2048"))
    
    # Qwen3 Rerank 模型配置
    @property
    def QWEN3_RERANKER_MODEL_BASE_URL(self) -> str:
        return os.getenv("QWEN3_RERANKER_MODEL_BASE_URL", "http://localhost:9903/rerank")
    
    # Jina Rerank 模型配置
    @property
    def JINA_RERANKER_MODEL_BASE_URL(self) -> str:
        return os.getenv("JINA_RERANKER_MODEL_BASE_URL", "http://localhost:9907/v1/rerank")
    
    # 小红书 OCR 模型配置
    @property
    def DOTS_OCR_MODEL_BASE_URL(self) -> str:
        return os.getenv("DOTS_OCR_MODEL_BASE_URL", "http://localhost:9906/v1")
    
    # MinerU VLM 模型配置
    @property
    def MINERU_VLM_SERVER_URL(self) -> str:
        return os.getenv("MINERU_VLM_SERVER_URL", "http://localhost:9909")
    
    # Milvus 向量数据库配置
    @property
    def VECTOR_DATABASE_URI(self) -> str:
        return os.getenv("VECTOR_DATABASE_URI", "http://localhost:19530")
    
    @property
    def VECTOR_DATABASE_NAME(self) -> str:
        return os.getenv("VECTOR_DATABASE_NAME", "steins")
    
    @property
    def VECTOR_COLLECTION_NAME(self) -> str:
        return os.getenv("VECTOR_COLLECTION_NAME", "aigc")
    
    @property
    def VECTOR_DATABASE_CHUNK_SIZE(self) -> int:
        return int(os.getenv("VECTOR_DATABASE_CHUNK_SIZE", "12800"))
    
    # Neo4j 图数据库配置
    @property
    def NEO4J_BASE_URL(self) -> str:
        return os.getenv("NEO4J_BASE_URL", "neo4j://localhost:7688")
    
    @property
    def NEO4J_USERNAME(self) -> str:
        return os.getenv("NEO4J_USERNAME", "neo4j")
    
    @property
    def NEO4J_PASSWORD(self) -> str:
        return os.getenv("NEO4J_PASSWORD", "password")
    
    @property
    def NEO4J_DATABASE(self) -> str:
        return os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Redis 数据库配置
    @property
    def REDIS_HOST(self) -> str:
        return os.getenv("REDIS_HOST", "localhost")
    
    @property
    def REDIS_PORT(self) -> int:
        return int(os.getenv("REDIS_PORT", "6379"))
    
    @property
    def REDIS_PASSWORD(self) -> str:
        return os.getenv("REDIS_PASSWORD", "123456")
    
    @property
    def REDIS_DATABASE(self) -> int:
        return int(os.getenv("REDIS_DATABASE", "0"))
    
    @property
    def QWEN3_VL_EMBEDDING_PATH(self) -> str:
        return os.getenv("QWEN3_VL_EMBEDDING_PATH", "EMPTY")
    
    @property
    def QWEN3_VL_RERANKER_PATH(self) -> str:
        return os.getenv("QWEN3_VL_RERANKER_PATH", "EMPTY")


# 创建全局配置实例
settings = Settings()