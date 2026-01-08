"""
VisualReaderTool 单元测试
测试 visual_reader_tool.py 中的 read_page 方法
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.code.Tools.visual_reader_tool import VisualReaderTool


class TestVisualReaderTool:
    """VisualReaderTool 类的单元测试"""
    
    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """创建一个临时 PDF 文件路径"""
        return str(tmp_path / "test_document.pdf")
    
    @pytest.fixture
    def mock_vision_model(self):
        """创建模拟的视觉模型"""
        mock_model = Mock()
        mock_model.run = Mock()
        return mock_model
    
    @pytest.fixture
    def mock_images(self):
        """创建模拟的图像列表"""
        images = []
        for i in range(2):
            img = Mock(spec=Image.Image)
            img.size = (800, 600)
            images.append(img)
        return images
    
    @pytest.fixture
    def tool_instance(self, sample_pdf_path, mock_vision_model):
        """创建 VisualReaderTool 实例，使用模拟的视觉模型"""
        with patch('src.code.Tools.visual_reader_tool.ModelFactory') as mock_factory:
            mock_factory.create.return_value = mock_vision_model
            tool = VisualReaderTool(pdf_path=sample_pdf_path)
            return tool

    def test_read_page_invalid_pdf_path(self):
        """测试无效的 PDF 路径应抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError) as exc_info:
            VisualReaderTool(pdf_path="")
        assert "必须提供 PDF 文档路径" in str(exc_info.value)

    def test_read_page_empty_pdf_path(self, sample_pdf_path):
        """测试空字符串 PDF 路径"""
        with pytest.raises(FileNotFoundError):
            VisualReaderTool(pdf_path="")

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    def test_get_page_image_success(self, mock_convert, tool_instance, mock_images):
        """测试 _get_page_image 成功获取图像"""
        mock_convert.return_value = mock_images
        
        result = tool_instance._get_page_image(1, 2)
        
        assert result == mock_images
        mock_convert.assert_called_once_with(
            tool_instance.pdf_path, first_page=1, last_page=2
        )

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    def test_get_page_image_no_images(self, mock_convert, tool_instance):
        """测试 _get_page_image 未获取到图像时返回 None"""
        mock_convert.return_value = []
        
        result = tool_instance._get_page_image(1, 2)
        
        assert result is None

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    def test_get_page_image_exception(self, mock_convert, tool_instance):
        """测试 _get_page_image 异常处理"""
        import pdf2image
        mock_convert.side_effect = Exception("PDF processing error")
        
        with pytest.raises(Exception) as exc_info:
            tool_instance._get_page_image(1, 2)
        assert "PDF processing error" in str(exc_info.value)

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    def test_read_page_no_images_returns_error_message(
        self, mock_convert, tool_instance, mock_images
    ):
        """测试无法获取页面图像时返回错误消息"""
        mock_convert.return_value = []
        
        result = tool_instance.read_page(
            page_indexes=(1, 2), 
            focus_query="请总结这一页的内容"
        )
        
        expected = "无法获取页面图像，无法回答问题。"
        assert result == expected

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    @patch('src.code.Tools.visual_reader_tool.ChatAgent')
    @patch('src.code.Tools.visual_reader_tool.BaseMessage')
    def test_read_page_success(
        self, mock_base_message, mock_chat_agent, 
        mock_convert, tool_instance, mock_images
    ):
        """测试 read_page 成功执行并返回格式化结果"""
        mock_convert.return_value = mock_images
        
        # 模拟 BaseMessage
        mock_assistant_msg = Mock()
        mock_user_msg = Mock()
        mock_base_message.make_assistant_message.return_value = mock_assistant_msg
        mock_base_message.make_user_message.return_value = mock_user_msg
        
        # 模拟 ChatAgent
        mock_agent_instance = Mock()
        mock_answer = Mock()
        mock_answer.msg.content = "这是模型生成的答案内容"
        mock_agent_instance.step.return_value = mock_answer
        mock_chat_agent.return_value = mock_agent_instance
        
        result = tool_instance.read_page(
            page_indexes=(5, 6), 
            focus_query="这一页讲了什么？"
        )
        
        # 验证返回结果格式
        assert "根据第 5 到 6 页的内容，回答如下：" in result
        assert "这是模型生成的答案内容" in result
        
        # 验证 BaseMessage 被正确调用
        mock_base_message.make_assistant_message.assert_called_once()
        call_args = mock_base_message.make_user_message.call_args
        assert "请阅读以下页面内容，并回答我的问题：这一页讲了什么？" in call_args[1]['content']

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    @patch('src.code.Tools.visual_reader_tool.ChatAgent')
    @patch('src.code.Tools.visual_reader_tool.BaseMessage')
    def test_read_page_system_message_content(
        self, mock_base_message, mock_chat_agent, 
        mock_convert, tool_instance, mock_images
    ):
        """测试 read_page 系统消息包含正确的指令"""
        mock_convert.return_value = mock_images
        
        mock_assistant_msg = Mock()
        mock_user_msg = Mock()
        mock_base_message.make_assistant_message.return_value = mock_assistant_msg
        mock_base_message.make_user_message.return_value = mock_user_msg
        
        mock_agent_instance = Mock()
        mock_answer = Mock()
        mock_answer.msg.content = "test"
        mock_agent_instance.step.return_value = mock_answer
        mock_chat_agent.return_value = mock_agent_instance
        
        tool_instance.read_page((1, 1), "test")
        
        # 验证系统消息内容包含关键指令
        sys_msg_call = mock_base_message.make_assistant_message.call_args
        sys_msg_content = sys_msg_call[1]['content']
        
        assert "视觉解析助手" in sys_msg_content
        assert "Markdown" in sys_msg_content
        assert "表格" in sys_msg_content or "流程图" in sys_msg_content

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    @patch('src.code.Tools.visual_reader_tool.ChatAgent')
    @patch('src.code.Tools.visual_reader_tool.BaseMessage')
    def test_read_page_agent_initialization(
        self, mock_base_message, mock_chat_agent, 
        mock_convert, tool_instance, mock_images
    ):
        """测试 read_page 中 ChatAgent 正确初始化"""
        mock_convert.return_value = mock_images
        
        mock_base_message.make_assistant_message.return_value = Mock()
        mock_base_message.make_user_message.return_value = Mock()
        
        mock_chat_agent.return_value = Mock()
        mock_chat_agent.return_value.step.return_value = Mock(msg=Mock(content=""))
        
        tool_instance.read_page((1, 1), "test")
        
        # 验证 ChatAgent 被调用，验证 message_window_size=5
        mock_chat_agent.assert_called_once()
        call_kwargs = mock_chat_agent.call_args[1]
        assert call_kwargs['message_window_size'] == 5
        assert call_kwargs['model'] == tool_instance.vision_model

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    @patch('src.code.Tools.visual_reader_tool.ChatAgent')
    @patch('src.code.Tools.visual_reader_tool.BaseMessage')
    def test_read_page_various_page_ranges(
        self, mock_base_message, mock_chat_agent, 
        mock_convert, tool_instance, mock_images
    ):
        """测试 read_page 处理不同的页码范围"""
        mock_convert.return_value = mock_images
        
        mock_base_message.make_assistant_message.return_value = Mock()
        mock_base_message.make_user_message.return_value = Mock()
        
        mock_agent = Mock()
        mock_agent.step.return_value = Mock(msg=Mock(content="答案"))
        mock_chat_agent.return_value = mock_agent
        
        # 测试单页
        result_single = tool_instance.read_page((1, 1), "单页问题")
        assert "根据第 1 到 1 页的内容" in result_single
        
        # 测试多页
        result_multi = tool_instance.read_page((10, 15), "多页问题")
        assert "根据第 10 到 15 页的内容" in result_multi

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    @patch('src.code.Tools.visual_reader_tool.ChatAgent')
    @patch('src.code.Tools.visual_reader_tool.BaseMessage')
    def test_read_page_user_msg_includes_images(
        self, mock_base_message, mock_chat_agent, 
        mock_convert, tool_instance, mock_images
    ):
        """测试 read_page 用户消息包含图像列表"""
        mock_convert.return_value = mock_images
        
        mock_base_message.make_assistant_message.return_value = Mock()
        mock_user_msg = Mock()
        mock_base_message.make_user_message.return_value = mock_user_msg
        
        mock_chat_agent.return_value = Mock()
        mock_chat_agent.return_value.step.return_value = Mock(msg=Mock(content=""))
        
        tool_instance.read_page((1, 2), "test")
        
        # 验证 image_list 参数被传递
        call_kwargs = mock_base_message.make_user_message.call_args[1]
        assert 'image_list' in call_kwargs
        assert call_kwargs['image_list'] == mock_images


class TestVisualReaderToolEdgeCases:
    """VisualReaderTool 边缘情况测试"""
    
    @pytest.fixture
    def pdf_path(self, tmp_path):
        return str(tmp_path / "test.pdf")
    
    @pytest.fixture
    def mock_vision_model(self):
        return Mock()
    
    @pytest.fixture
    def tool(self, pdf_path, mock_vision_model):
        with patch('src.code.Tools.visual_reader_tool.ModelFactory') as mock_factory:
            mock_factory.create.return_value = mock_vision_model
            return VisualReaderTool(pdf_path=pdf_path)

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    @patch('src.code.Tools.visual_reader_tool.ChatAgent')
    @patch('src.code.Tools.visual_reader_tool.BaseMessage')
    def test_read_page_empty_query(self, mock_base_message, mock_chat_agent, 
                                   mock_convert, tool, mock_vision_model):
        """测试 read_page 处理空查询字符串"""
        mock_convert.return_value = [Mock()]
        
        mock_base_message.make_assistant_message.return_value = Mock()
        mock_base_message.make_user_message.return_value = Mock()
        
        mock_chat_agent.return_value = Mock()
        mock_chat_agent.return_value.step.return_value = Mock(msg=Mock(content=""))
        
        result = tool.read_page((1, 1), "")
        
        # 验证消息仍然被创建
        mock_base_message.make_user_message.assert_called_once()

    @patch('src.code.Tools.visual_reader_tool.convert_from_path')
    @patch('src.code.Tools.visual_reader_tool.ChatAgent')
    @patch('src.code.Tools.visual_reader_tool.BaseMessage')
    def test_read_page_long_query(self, mock_base_message, mock_chat_agent, 
                                  mock_convert, tool, mock_vision_model):
        """测试 read_page 处理长查询字符串"""
        long_query = "a" * 1000
        mock_convert.return_value = [Mock()]
        
        mock_base_message.make_assistant_message.return_value = Mock()
        mock_base_message.make_user_message.return_value = Mock()
        
        mock_chat_agent.return_value = Mock()
        mock_chat_agent.return_value.step.return_value = Mock(msg=Mock(content=""))
        
        result = tool.read_page((1, 1), long_query)
        
        # 验证长查询被包含在消息中
        call_kwargs = mock_base_message.make_user_message.call_args[1]
        assert long_query in call_kwargs['content']