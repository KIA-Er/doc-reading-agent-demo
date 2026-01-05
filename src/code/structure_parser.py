"""
code.src.structure_parser 的 Docstring
本脚本用于提取Word标题预计对应页码范围的结构化信息
"""
import fitz  # PyMuPDF
from docx import Document
from loguru import logger
from typing import List, Dict, Any
import os

from pathlib import Path
#当前脚本所在目录
HERE = Path(__file__).resolve().parent
#项目根目录
ROOT = HERE.parent.parent

#配置日志到控制台
logger = logger.bind(module="structure_parser")

# 导入标题提取器
from .title_extractor import TitleExtractor

def parse_structure(doc_path: str = f"{ROOT}/示例数据/test.docx", pdf_path: str= f"{ROOT}/示例数据/test.pdf") -> List[Dict[str, Any]]:
    """
    解析Word文档的结构信息
    Args:
        doc_path (str): Word文档的路径
        pdf_path (str): 对应的PDF文档路径
    Returns:
        List[Dict[str, Any]]: 包含标题及其页码范围的结构
        示例：
        [
            {
                "title": "第一章 绪论",
                "start_page": 1,
                "end_page": 5
            },
            {
                "title": "第二章 文献综述",
                "start_page": 6,
                "end_page": 15
            }
        ]
    """
    #检查传入参数是否完整if not doc_path or not pdf_path:
    try:
        if not doc_path or not pdf_path:
            logger.error("必须提供 Word 文档路径和对应的 PDF 路径")
            return []
    except Exception as e:
        logger.error(f"检查路径参数时出错: {e}")
        return []

    # 初始化结构信息列表
    structure_info = []

    if not os.path.exists(doc_path):
        logger.error(f"文件不存在: {doc_path}")
        return structure_info
    # 读取Word文档
    try:
        doc = Document(doc_path)
    except Exception as e:
        logger.error(f"读取Word文档时出错: {e}")
        return structure_info

    # 提取标题信息
    headings = extract_headings(doc)
    if not headings:
        logger.warning("未在文档中检测到标题样式的段落")
        return structure_info
    
    # 将Word转换为PDF以获取页码信息（或者通过已有的PDF文件）
    pdf_doc = fitz.open(pdf_path)
    # 搜索游标，记录上一个标题所在的页码
    current_cursor = 0
    # 存储标题及其起始页码
    heading_pages = []

    for heading in headings:
            # 1. 从游标位置开始往后搜
            found_page = get_page_number_for_heading(pdf_doc, heading, start_page=current_cursor)
            
            # 2. 如果没找到（可能是 Word 排版导致的换行问题），尝试模糊搜索
            if found_page == -1 and len(heading) > 10:
                 # 只搜前 15 个字
                short_text = heading[:15]
                found_page = get_page_number_for_heading(pdf_doc, short_text, start_page=current_cursor)

            if found_page != -1:
                logger.info(f"  📍 [P.{found_page}] {heading}")
                heading_pages.append((heading, found_page))
                # 【关键】更新游标：下一个标题不可能出现在当前标题之前
                # 所以下次搜索直接从当前页开始
                current_cursor = found_page
            else:
                logger.warning(f"  ❌ [未找到] {heading}")
                # 如果没找到，游标不更新，保持在原地，继续找下一个标题

    for idx, (heading, start_page) in enumerate(heading_pages):
        # 下一个标题的页码用于推断当前标题的结束页，至少不小于当前起始页
        if idx + 1 < len(heading_pages):
            next_start = heading_pages[idx + 1][1]
            end_page = max(start_page, next_start - 1)
        else:
            end_page = start_page  # 最后一个标题暂且认为覆盖到文件结束

        structure_info.append(
            {
                "title": heading,
                "start_page": start_page,
                "end_page": end_page
            }
        )

    return structure_info

def extract_headings(doc: Document, min_score: float = 60.0) -> List[str]:
    """提取Word文档中的标题文本。
    
    基于多特征评分系统智能识别标题，不依赖样式名称。
    使用TitleExtractor实现，支持多级标题识别和可配置的评分阈值。
    
    Args:
        doc: Word文档对象
        min_score: 标题最低评分阈值，默认60.0分
        
    Returns:
        List[str]: 提取的标题文本列表
    """
    # 初始化标题提取器
    extractor = TitleExtractor()
    
    # 提取标题，使用指定的评分阈值
    titles = extractor.extract(doc, min_score=min_score)
    
    # 提取标题文本，保持与原有函数接口兼容
    headings = [title['text'] for title in titles]
    
    logger.info(f"使用评分系统提取到 {len(headings)} 个标题（阈值: {min_score}）")
    
    # 统计各级标题数量
    level_counts = {}
    for title in titles:
        level = title['level']
        level_counts[level] = level_counts.get(level, 0) + 1
    
    if level_counts:
        logger.debug(f"标题级别分布: {level_counts}")
    
    return headings

    """
    获取标题所在的页码
    Args:
        heading (str): 标题文本
    Returns:
        int: 标题所在的页码
    """
    # 这里需要实现具体的逻辑来获取标题所在页码
    # 可能需要结合PDF解析库如PyMuPDF来实现
    return 1  # 示例返回第一页

def get_page_number_for_heading(pdf_doc: fitz.Document, target_text: str, start_page: int = 0) -> int:
        """
        轻量化搜索：只从 start_page 开始往后找
        """
        clean_text = target_text.strip()
        if not clean_text:
            return -1

        total_pages = len(pdf_doc)
        
        # 从 start_page 开始，直到文档结束
        for i in range(start_page, total_pages):
            page = pdf_doc[i]
            # hit_max=1: 只要找到一处就立马返回，不再扫描整页其他位置
            if page.search_for(clean_text):
                return i
        
        return -1


def main():
    docs_path = ROOT/"示例数据/test.docx"
    pdf_path = ROOT/"示例数据/test.pdf"
    structure = parse_structure(docs_path, pdf_path)
    for item in structure:
        logger.info(f"标题: {item['title']}, 起始页: {item['start_page']}, 结束页: {item['end_page']}")

if __name__ == "__main__":
    main()