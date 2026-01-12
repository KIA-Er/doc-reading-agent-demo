import re
from typing import List, Dict, Any
from docx import Document
from docx.text.paragraph import Paragraph
from loguru import logger  # 保持你原有的 logger

class TitleExtractor:
    """
    智能标题提取器 (v2.0 规则优化版)
    使用 '正则模式 + 显著性特征' 双重校验，替代旧的评分系统。
    """
    
    def __init__(self):
        # 定义正则模式: (正则对象, 标题级别, 是否为弱模式)
        # 弱模式(True)意味着必须配合加粗或大字号才能算标题
        self.patterns = [
            # Level 1: "第一章", "第六章" (强模式)
            (re.compile(r'^第[一二三四五六七八九十0-9]+[章节]'), 'chapter', False),
            
            # Level 2: "格式二：", "格式三" (强模式 - 解决漏判)
            (re.compile(r'^格式[一二三四五六七八九十0-9]+[：:]?'), 'level2', False),
            
            # Level 2: "一、", "二、" (强模式)
            (re.compile(r'^[一二三四五六七八九十]+、'), 'level2', False),
            
            # Level 3: "1.1", "1.2.3" (中等模式 - 只要符合且不长通常就是标题)
            (re.compile(r'^\d+(\.\d+)+'), 'level3', False),
            
            # Level 4: "5.", "1." (弱模式 - 极易误判，需要校验样式)
            (re.compile(r'^\d+[.、]'), 'level4', True), 
        ]

    def _is_emphasized(self, para: Paragraph) -> bool:
        """判断段落是否有'显著性'特征 (加粗 或 字号较大)"""
        # 1. 检查 Style 级别的加粗
        if para.style.font.bold:
            return True
        
        max_size = 0
        has_bold_run = False
        
        # 2. 检查 Run 级别的加粗和字号
        if para.runs:
            for run in para.runs:
                if run.bold:
                    has_bold_run = True
                if run.font.size:
                    max_size = max(max_size, run.font.size)
        
        if has_bold_run:
            return True

        # 3. 检查字号
        # 160000 EMUs ≈ 12.5pt (大于小四号)。
        # 一般正文是五号(10.5pt/133333 EMUs)，标题通常大于它。
        # 如果 max_size 为 0 (None)，通常意味着继承正文样式，视为不显著。
        if max_size > 160000: 
            return True
            
        return False

    def extract(self, doc: Document, min_score: float = 60.0) -> List[Dict[str, Any]]:
        """
        执行提取逻辑
        
        Args:
            doc: Word文档对象
            min_score: (兼容性保留) 在规则模式下暂不使用此参数
        """
        titles = []
        
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if not text:
                continue

            # --- 1. 基础清洗 (Veto Rules) ---
            
            # 长度过滤: 标题通常很短 (放宽到 50 以适应 "关于...的说明")
            if len(text) > 50: 
                continue
                
            # 标点过滤: 绝对不以句号、分号结尾 (解决条款误判的核心)
            if text.endswith(('。', '.', '；', ';')):
                continue
            
            # 逗号过滤: 标题里很少出现逗号
            if ('，' in text or ',' in text):
                continue

            # --- 2. 模式匹配 ---
            for pattern, level, is_weak in self.patterns:
                if pattern.match(text):
                    
                    # 【核心优化】: 弱模式必须校验显著性
                    # 解决 "5.信用记录查询" 这种正文列表项的问题
                    if is_weak:
                        if not self._is_emphasized(para):
                            continue # 长得像标题但太小了，跳过
                    
                    # 匹配成功，加入结果
                    # 这里的 score 给 100 是为了兼容旧逻辑的结构
                    titles.append({
                        'index': i,
                        'text': text,
                        'score': 100.0, 
                        'level': level,
                        'details': {'type': 'rule_match'}
                    })
                    break # 匹配到一个模式就停止，防止重复

        return titles