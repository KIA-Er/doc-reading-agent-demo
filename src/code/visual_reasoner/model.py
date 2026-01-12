from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.messages import BaseMessage
from camel.agents.chat_agent import ChatAgent
from typing import List, Any, Dict, Optional
from PIL import Image, ImageDraw, ImageFont
import os

from src.settings import settings
from src.code.rerank.reranker import Reranker
from loguru import logger
from src.code.data_base.database import VectorDatabase, vector_db
import asyncio

logger = logger.bind(module="visual_reasoner_model")


class VisionLanguageModel:
    def __init__(
            self,
            model_platform: ModelPlatformType = ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_name: str = settings.VLM_MODEL_NAME,
            url: str = settings.VLM_BASE_URL,
            ):

        self.vison_model = ModelFactory.create(
            model_platform=model_platform,
            model_type=model_name,
            url=url,
            model_config_dict={
                "temperature": 0.1,
            }
        )
        self.database = vector_db
        logger.info(f"VisionLanguageModel 已就绪")

    def run(self,query:str, image_urls: List[str]):

        images = self._load_images_from_urls(image_urls)
        images_pages = [self.database.get_page_index_by_image_url(image_url=image_url) for image_url in image_urls]

        """
        原始代码：
        
        vision_sys_msg = BaseMessage.make_assistant_message(
                role_name="VisionEye",
                content=(
                    "你是一个视觉解析助手。请根据用户的 query 详细理解页面内容并进行回答。"
                    "如果是表格，请还原结构；如果是流程图，请描述流转步骤。"
                    "只输出有来源依据的客观事实，不要添加无关的开场白。"
                    "务必使用 Markdown 格式进行回答。"
                    "必须回答与用户 query 相关的内容并显式指定你回答的来源页面，如：来源第 1 页。"
                )
            )

        agent = ChatAgent(
            system_message=vision_sys_msg,
            model= self.vison_model,
            message_window_size=5,
        )

        page_mapping_desc = []
        for i, page_idx in enumerate(images_pages):
            page_mapping_desc.append(f"- 第 {i+1} 张图片对应文档的【第 {page_idx} 页】")

        page_mapping_desc = "\n".join(page_mapping_desc)

        print(page_mapping_desc)
        logger.info(f"页面索引映射关系:\n{page_mapping_desc}")

        final_prompt = (
            f"请仔细阅读随附的 {len(images)} 张图片，并回答：{query}。\n\n"
            f"### 页面索引映射关系（重要）：\n{page_mapping_desc}\n\n"
            f"####要求（重要）：请在回答时根据上述映射关系，明确引用来源页码而不是图片序号。"
        )

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=final_prompt,#BUG:这里好像有bug，模型无法根据图片与传入的页面坐标对应上来。
            image_list=images,#List[Image]
        )"""
        
        """
        添加视觉水印检查模型是否可以正确对应页面
        """
        processed_images = []
        for img, p_idx in zip(images, images_pages):
            # 给图片打上 "Page 36" 的水印
            new_img = self._add_page_number_to_image(img, p_idx)
            processed_images.append(new_img)
        
        vision_sys_msg = BaseMessage.make_assistant_message(
            role_name="VisionEye",
            content=(
                "你是一个精准的文档视觉分析助手。你的任务是根据用户问题从图片中提取答案。\n"
                "### 核心规则：\n"
                "1. **视觉锚点**：每张图片的**左上角**都有一个红色的页码标记（例如 '|<Page 1>|'）。\n"  # <--- 关键修改：告诉它看哪里
                "2. **来源引用**：在回答时，**必须**直接引用该视觉标记上的页码。例如：'根据 |<Page 1>| 的内容...'。\n"
                "3. **客观陈述**：如果是表格，请还原结构；如果是流程图，请描述流转步骤。\n"
                "4. **格式要求**：使用 Markdown 格式。"
            )
        )
        agent = ChatAgent(
            system_message=vision_sys_msg,
            model= self.vison_model,
            message_window_size=5,
        )
        final_prompt = (
            f"请阅读随附的 {len(images)} 张图片，回答问题：【{query}】。\n\n"
            f"⚠️ **重要提示**：\n"
            f"- 我已在每张图片的左上角标注了真实页码（如 |<Page 12>|, |<Page 36>|）。\n"
            f"- 请**忽略**图片在列表中的顺序，**只认图片上印着的页码数字**。\n"
            f"- 如果某张图没有包含问题的答案，请直接忽略该图。"
        )

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=final_prompt,
            image_list=processed_images,
        )
        answer = agent.step(user_msg)
        content = answer.msg.content
        return content
    
    def _load_images_from_urls(self, image_urls: List[str]) -> List[Any]:

        loaded_images = []

        for path in image_urls:
            if not os.path.exists(path):
                print(f"⚠️ 警告: 文件不存在，跳过: {path}")
                continue
                
            try:
                img = Image.open(path)
                
                # 因为 PIL 默认是懒加载，不 .load() 的话文件句柄会一直开着
                img.load()
                
                loaded_images.append(img)
                
            except Exception as e:
                print(f"❌ 错误: 无法加载图片 {path}: {e}")
                continue
            
        return loaded_images
    
    def _add_page_number_to_image(self, image: Image.Image, page_num: int) -> Image.Image:
        """在图片左上角强行绘制页码水印"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        text = f"|<Page {page_num}>|"
        
        # 如果没有字体文件，默认字体可能太小，建议加载个 ttf
        draw.text((10, 10), text, fill="red", font_size=50) # 这里用红色，方便看到
        # 为了演示简单，这里用默认字体，实际建议放大
        try:
            # 尝试加载大字体
            font = ImageFont.truetype("arial.ttf", 60)
        except:
            font = None # 回退默认
            
        # 画一个白底背景框，防止文字看不清
        draw.rectangle([0, 0, 300, 80], fill="white")
        draw.text((10, 10), text, fill="red", font=font)
        
        return img_copy
    

if __name__ == "__main__":
    logger.disable("src.code.embedding")
    logger.disable("src.code.rerank")

    query = """关于中小微企业投标，我要注意是什么？"""    
    reranker = Reranker(return_documents=True)
    response = asyncio.run(reranker.rerank(
        query=query, 
        img_urls=[f"/mnt/ssd2/steins/wenkai/project/doc-reading-agent-demo/demo_data_images/test_{i}.jpeg"for i in range(1,67)]
        ))
    image_urls = [ item['document']['text'] for item in response['results']]
    # print(image_urls)

    model = VisionLanguageModel()
    response = model.run(
        query=query,
        image_urls=image_urls
    )
    print(response)