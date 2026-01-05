import re
from typing import List, Dict, Any
from docx import Document

def extract_lightweight_headings(docx_path: str) -> List[Dict[str, Any]]:
    """
    极简标题提取器 (针对无样式文档)
    策略：严格限制长度 + 标点符号过滤 + 正则匹配
    """
    doc = Document(docx_path)
    headings = []

    # 定义正则模式 (按优先级排序)
    patterns = [
        # Level 1: "第一章", "第1章"
        (re.compile(r'^第[一二三四五六七八九十0-9]+[章节]'), 1),
        # Level 2: "一、", "二、"
        (re.compile(r'^[一二三四五六七八九十]+、'), 2),
        # Level 3: "1.1", "1.2.3"
        (re.compile(r'^\d+(\.\d+)+'), 3),
        # Level 4: "1.", "2." (风险最高，需严格限制)
        (re.compile(r'^\d+[.、]'), 4),
    ]

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # --- 核心过滤规则 (Veto Rules) ---
        
        # 1. 【长度过滤】: 标题通常很短。超过 30 个字绝对是正文条款。
        #    (解决了你日志里 "1.乙方提供的服务..." 的问题)
        if len(text) > 30: 
            continue

        # 2. 【标点过滤】: 标题结尾不应该有句号、分号、冒号。
        if text.endswith(('。', '.', '；', ';', '：', ':')):
            continue

        # 3. 【逗号过滤】: 标题里很少出现逗号（除非是副标题，但POC可忽略）
        if '，' in text or ',' in text:
            continue

        # --- 匹配逻辑 ---
        for pattern, level in patterns:
            if pattern.match(text):
                # 额外检查：如果是纯数字开头(Level 4)，必须有加粗才算标题？
                # 这里为了轻量化，暂不检查加粗，依靠上面的长度过滤通常足够了
                
                headings.append({
                    "text": text,
                    "level": level
                })
                break # 匹配到一个模式就停止

    return headings

# --- 使用示例 ---
if __name__ == "__main__":
    # 替换为你的文件路径
    DOC_PATH = "示例数据/test.docx" 
    
    results = extract_lightweight_headings(DOC_PATH)
    
    print(f"提取到 {len(results)} 个标题:")
    for h in results:
        print(f"[Level {h['level']}] {h['text']}")