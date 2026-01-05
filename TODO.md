# Visual Document Agent - 开发进度表

## 🛠️ Phase 1: 环境准备
- [x] **T1.1**: `uv init` 及依赖安装 (camel-ai, python-docx, pymupdf, pdf2image)
- [x] **T1.2**: 准备测试数据 (spec.docx, spec.pdf) 并放入根目录

## 🗺️ Phase 2: 结构化解析器 (The Map)
- [ ] **T2.1**: 实现 `structure_parser.py` - 提取 Word 标题
- [ ] **T2.2**: 实现 PDF 页码定位逻辑 - 生成 Map JSON
- [ ] **验证**: 运行解析脚本，确认章节页码对应正确

## 👁️ Phase 3: 视觉工具 (The Eyes)
- [ ] **T3.1**: 实现 `vision_tools.py` - PDF 特定页转图片
- [ ] **T3.2**: 集成 Camel GPT-4o - 实现“看图说话”函数
- [ ] **验证**: 手动调用函数读取第 5 页，检查描述是否准确

## 🧠 Phase 4: Agent 集成 (The Brain)
- [ ] **T4.1**: 将上述功能封装为 `FunctionTool`
- [ ] **T4.2**: 编写 `System Prompt` (策略：先看目录 -> 后看内容)
- [ ] **T4.3**: 编写 `main.py` 入口与测试流程
- [ ] **最终验证**: 从头到尾跑通 "查询 xxx 流程" 的用例