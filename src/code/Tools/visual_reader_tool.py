from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent

from typing import Callable, List
from src.settings import settings
from loguru import logger
from PIL import Image
from pdf2image import convert_from_path
from io import BytesIO

from camel.toolkits import FunctionTool
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

logger = logger.bind(module="visual_reader_tool")

class VisualReaderTool:

    def __init__(self):

        self.vision_model = ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            model_type=settings.VLM_MODEL_NAME,
            url=settings.VLM_BASE_URL,
            model_config_dict={
                "temperature": 0.0,
            }
        )
        logger.info(f"VisualReaderTool 已就绪")

    def _get_page_image(self, image_path: str, first_page: int, last_page: int) -> List[Image.Image]:
        try:
            images = convert_from_path(image_path, first_page=first_page, last_page=last_page)
            
            if not images:
                logger.error(f"转换PDF页面为图像时未获取到任何图像: {image_path} 第 {first_page}-{last_page} 页")
                return [] 

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

        except Exception as e:
            logger.error(f"PDF转图片异常: {e}")
            raise

    def read_page(self, image_path: str, page_indexes: tuple, focus_query: str) -> str:
        """
        利用视觉语言模型读取指定页码范围的内容并回答问题
        Args:
            page_indexes (tuple): 包含起始页码和结束页码的元组 (first_page, last_page)
            focus_query (str): 用户的问题或关注点
        Returns:
            str: 页面内容的详细文本描述（Markdown格式）。
        """
        images = self._get_page_image(image_path, page_indexes[0], page_indexes[1])
        if not images:
            return "无法获取页面图像，无法回答问题。"

        vision_sys_msg = BaseMessage.make_assistant_message(
                role_name="VisionEye",
                content=(
                    "你是一个视觉解析助手。请根据用户的 query 详细描述图片内容。"
                    "如果是表格，请还原结构；如果是流程图，请描述流转步骤。"
                    "只输出客观事实，不要添加无关的开场白。"
                    "请使用 Markdown 格式进行回答。"
                )
            )
        
        agent = ChatAgent(
            system_message=vision_sys_msg,
            model=self.vision_model,
            message_window_size=5,
        )   

        user_msg = BaseMessage.make_user_message(
            role_name="User",
            content=f"请阅读以下页面内容，并回答我的问题：{focus_query}",
            image_list=images,
        )
        answer = agent.step(user_msg)
        content = answer.msg.content
        return f"根据第 {page_indexes[0]} 到 {page_indexes[1]} 页的内容，回答如下：\n\n{content}"

    
    
def get_visual_reader_tool()->FunctionTool:
    tool_instance = VisualReaderTool()
    return FunctionTool(tool_instance.read_page)

# image_path = ROOT/"示例数据/test.pdf"
# tool = VisualReaderTool()
# answer = tool.read_page(str(image_path), (12,13),"免疫组织化学染色八项（手工法）的收费代码有哪些？")
# print(answer)

