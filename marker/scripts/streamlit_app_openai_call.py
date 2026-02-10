import os

from marker.scripts.common import (
    load_models,
    parse_args,
    img_to_html,
    get_page_image,
    page_count,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["IN_STREAMLIT"] = "true"

from marker.settings import settings
from streamlit.runtime.uploaded_file_manager import UploadedFile

import re
import tempfile
from typing import Any, Dict

import streamlit as st
from PIL import Image

from marker.converters.pdf import PdfConverter
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered
# from marker.config.apikey import LLMAPI_KEY
from openai import OpenAI
from streamlit_ace import st_ace
import io
import base64
import oss2
import uuid

def convert_pdf(fname: str, config_parser: ConfigParser) -> (str, Dict[str, Any], dict):
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1
    converter_cls = PdfConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=model_dict,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    return converter(fname)


# def markdown_insert_images(markdown, images):
#     image_tags = re.findall(
#         r'(!\[(?P<image_title>[^\]]*)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))',
#         markdown,
#     )

#     for image in image_tags:
#         image_markdown = image[0]
#         image_alt = image[1]
#         image_path = image[2]
#         if image_path in images:
#             markdown = markdown.replace(
#                 image_markdown, img_to_html(images[image_path], image_alt)
#             )
#     return markdown

# image upload to cloud and return url？？？

def img2cloud_to_html(img, img_alt):
    # 1. Get OSS configuration from environment variables
    access_key_id = os.environ.get("OSS_ACCESS_KEY_ID")
    access_key_secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
    bucket_name = os.environ.get("OSS_BUCKET_NAME")
    endpoint = os.environ.get("OSS_ENDPOINT")

    # Check if configuration is complete
    if not all([access_key_id, access_key_secret, bucket_name, endpoint]):
        # Fallback or error message if config is missing
        return f'<p style="color:red;">Error: OSS configuration missing. Please check environment variables.</p>'

    # 2. Convert PIL image to byte stream
    img_bytes = io.BytesIO()
    # Use the format defined in settings, default to PNG if not set
    img_format = settings.OUTPUT_IMAGE_FORMAT if hasattr(settings, 'OUTPUT_IMAGE_FORMAT') else 'PNG'
    img.save(img_bytes, format=img_format)
    img_bytes.seek(0) # Reset pointer to the beginning of the stream

    # 3. Generate a unique filename to avoid overwriting
    # Using UUID to ensure uniqueness, and organizing into a 'streamlit_images' folder
    file_ext = img_format.lower()
    unique_filename = f"screenshots/{uuid.uuid4().hex}.{file_ext}"

    # 4. Initialize OSS Bucket object
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    try:
        # 5. Upload the file to OSS
        # put_object automatically handles the stream upload
        bucket.put_object(unique_filename, img_bytes)

        # 6. Construct the public access URL
        # Remove protocol (http/https) from endpoint to avoid duplication
        clean_endpoint = endpoint.replace("https://", "").replace("http://", "")
        # Standard OSS URL format: https://{bucket-name}.{endpoint}/{filename}
        image_url = f"https://{bucket_name}.{clean_endpoint}/{unique_filename}"

        # 7. Return the HTML img tag
        url = f'<img src="{image_url}" alt="{img_alt}" style="max-width: 100%;">'
        print(url)
        return url

    except Exception as e:
        # Return error info in HTML if upload fails
        error = f'<p style="color:red;">Image Upload Failed: {str(e)}</p>'
        return error
def markdown_insert_images(markdown, images):
    image_tags = re.findall(
        r'(!\[(?P<image_title>[^\]]*)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))',
        markdown,
    )

    for image in image_tags:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if image_path in images:
            markdown = markdown.replace(
                image_markdown, img2cloud_to_html(images[image_path], image_alt)
            )
    return markdown

def remove_base64_images(text: str) -> str:
    """
    从文本中移除所有 <img> 标签（特别是包含 base64 数据的）
    支持单引号、双引号、无引号（不推荐但兼容）、跨行等场景
    """
    # 匹配 <img ...> 标签，特别针对 src="data:image/...base64,...
    pattern = r'<img\s+[^>]*src\s*=\s*["\']?data:image/[^"\'>]*["\']?[^>]*>'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned_text.strip()


def translate_with_openaicall_stream(text: str, target_language: str = "中文") -> str:
    base_url = "https://api.modelarts-maas.com/openai/v1"  # API地址
    # api_key = LLMAPI_KEY["MAAS_API_KEY"]  # 把MAAS_API_KEY替换成已获取的API Key 
    # export MAAS_API_KEY=*****
    api_key = os.environ.get("MAAS_API_KEY")
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        stream = client.chat.completions.create(
            model="qwen3-30b-a3b",
            messages=[
                {"role": "system", "content": f"请提供翻译的原材料内容,系统会将其翻译为中文并进行一定的整理，需求如下： \
                    1、图像标签的内容保持不变; \
                    2、将翻译后文中出现的'我们'用'论文中'进行替换； \
                    3、暂时将引文部分的链接去掉; \
                    4、将公式以latex的格式进行整理; \
                    5、整体结果以markdown格式输出。"},
                {"role": "user", "content": f"好的，内容如下:{text}。请帮助翻译整理和修订，谢谢。"}
            ],
            stream=True,
            temperature=0.7
        )

        # 逐块打印响应
        for chunk in stream:
            # 安全地检查并获取内容
            # if (hasattr(chunk, 'choices') and 
            #     chunk.choices and 
            #     len(chunk.choices) > 0 and
            #     hasattr(chunk.choices[0], 'delta') and
            #     chunk.choices[0].delta and
            #     hasattr(chunk.choices[0].delta, 'content') and
            #     chunk.choices[0].delta.content is not None):
            content = chunk.choices[0].delta.content
            if content:  # 确保内容不为空字符串
                yield content
    except Exception as e:
        yield f"API调用错误: {str(e)}"

st.set_page_config(layout="wide")
col1, col2 = st.columns([0.65, 0.35])

model_dict = load_models()
cli_options = parse_args()

# st.markdown("""
# # Marker Demo

# This app will let you try marker, a PDF or image -> Markdown, HTML, JSON converter. It works with any language, and extracts images, tables, equations, etc.

# Find the project [here](https://github.com/VikParuchuri/marker).
# """)

in_file: UploadedFile = st.sidebar.file_uploader(
    "PDF, document, or image file:",
    type=["pdf", "png", "jpg", "jpeg", "gif", "pptx", "docx", "xlsx", "html", "epub"],
)
editable_md = st.sidebar.checkbox("Enable editable markdown editor", value=False)
do_translate = st.sidebar.checkbox("Enable Translation", value=False)

output_format = st.sidebar.selectbox(
    "Output format", ["markdown", "json", "html", "chunks"], index=0
)

use_llm = st.sidebar.checkbox(
    "Use LLM", help="Use LLM for higher quality processing", value=False
)
force_ocr = st.sidebar.checkbox("Force OCR", help="Force OCR on all pages", value=False)
strip_existing_ocr = st.sidebar.checkbox(
    "Strip existing OCR",
    help="Strip existing OCR text from the PDF and re-OCR.",
    value=False,
)
debug = st.sidebar.checkbox("Debug", help="Show debug information", value=False)
disable_ocr_math = st.sidebar.checkbox(
    "Disable math",
    help="Disable math in OCR output - no inline math",
    value=False,
)

run_marker = st.sidebar.button("Run Marker")

if in_file is None:
    st.stop()

filetype = in_file.type

import fitz  # PyMuPDF
def pdf_to_images(pdf_file, dpi=150):
    """将上传的 PDF 文件转换为 PIL 图像列表"""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)  # 提高分辨率
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    pdf_document.close()
    return images

with col1:
    # 创建两行容器，比例为0.9:0.1
    # 获取总页数
    images = pdf_to_images(in_file)
    total_pages = len(images)
    
    # 第一行：显示图片和第一个页面选择器
    # 第一个页面选择器
    page_number = st.sidebar.number_input(
        f"Page number (top) out of {total_pages}:", 
        min_value=0, 
        max_value=total_pages-1,
        key="page_number"
    )

    # 显示对应页面的图片
    pil_image = images[page_number]
    st.image(pil_image, width='stretch')

page_range = st.sidebar.text_input(
    "Page range to parse, comma separated like 0,5-10,20",
    value=f"{page_number}-{page_number}",
)

if not run_marker:
    st.stop()

# Run Marker
with tempfile.TemporaryDirectory() as tmp_dir:
    temp_pdf = os.path.join(tmp_dir, "temp.pdf")
    with open(temp_pdf, "wb") as f:
        f.write(in_file.getvalue())

    cli_options.update(
        {
            "output_format": output_format,
            "page_range": page_range,
            "force_ocr": force_ocr,
            "debug": debug,
            "output_dir": settings.DEBUG_DATA_FOLDER if debug else None,
            "use_llm": use_llm,
            "strip_existing_ocr": strip_existing_ocr,
            "disable_ocr_math": disable_ocr_math,
        }
    )
    config_parser = ConfigParser(cli_options)
    rendered = convert_pdf(temp_pdf, config_parser)
    page_range = config_parser.generate_config_dict()["page_range"]
    first_page = page_range[0] if page_range else 0

#   if isinstance(rendered, MarkdownOutput):
#        return rendered.markdown, "md", rendered.images
text, ext, images = text_from_rendered(rendered)
with col2:
    if output_format == "markdown":
        text = markdown_insert_images(text, images)
        #st.markdown(text, unsafe_allow_html=True)
        # 侧栏开关：是否启用可编辑编辑器
        # 翻译选项
        # col_translate_left, col_translate_right = st.columns(2)
        # with col_translate_left:
        #     do_translate = st.checkbox("翻译为中文", value=False, key="translate_checkbox")
        # with col_translate_right:
                # 将 col2 分成上下两个区域
        col2_top = st.container()
        col2_bottom = st.container()

        with col2_top:
            st.subheader("📄 原始内容 (Original)")
            # st.markdown(text, unsafe_allow_html=True)
             
            edited_original = st_ace(
                value=text, 
                language="markdown", 
                theme="github", 
                height=300,
                key="original_editor") 
            display_original = text           
        # 翻译部分
        if do_translate:
            with col2_bottom:
                st.subheader("🌐 翻译内容 (Translated)")
                
                # 创建占位符用于流式显示翻译结果
                translation_placeholder = st.empty()
                translation_container = st.container()
                
                # 显示翻译进度
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 开始翻译
                status_text.info("正在翻译...")
                
                # 使用流式翻译
                translated_text = ""
                translation_display = ""
                
                try:
                    # 获取流式翻译结果
                    stream_generator = translate_with_openaicall_stream(display_original)
                    
                    # 逐步显示翻译结果
                    chunk_count = 0
                    for chunk in stream_generator:
                        translated_text += chunk
                        
                        # 使用 placeholder.markdown 进行替换更新，而不是追加
                        translation_placeholder.markdown(translated_text, unsafe_allow_html=True)
                
                        chunk_count += 1
                        progress = min(90, chunk_count * 5)  # 模拟进度到90%
                        progress_bar.progress(progress)
                
                    # 翻译完成
                    progress_bar.progress(100)
                    status_text.success("翻译完成！")
                    
                    # 最终显示完整翻译结果
                    with translation_container:
                        if editable_md:
                            # 可编辑的翻译结果
                            edited_translation = st_ace(
                                value=translated_text, 
                                language="markdown", 
                                theme="github", 
                                height=300,
                                key="translation_editor"
                            )
                            # 添加翻译结果下载按钮
                            st.download_button(
                                "下载翻译结果", 
                                data=edited_translation, 
                                file_name="translated_output.md", 
                                mime="text/markdown"
                            )
                        else:
                            st.markdown(translated_text, unsafe_allow_html=True)
                            # 添加翻译结果下载按钮
                            st.download_button(
                                "下载翻译结果", 
                                data=translated_text, 
                                file_name="translated_output.md", 
                                mime="text/markdown"
                            )
                
                except Exception as e:
                    status_text.error(f"翻译过程中出现错误: {str(e)}")
                    with translation_container:
                        st.error(f"翻译错误: {str(e)}")    
    elif output_format == "json":
        st.json(text)
    elif output_format == "html":
        st.html(text)
    elif output_format == "chunks":
        st.json(text)

if debug:
    with col1:
        debug_data_path = rendered.metadata.get("debug_data_path")
        if debug_data_path:
            pdf_image_path = os.path.join(debug_data_path, f"pdf_page_{first_page}.png")
            img = Image.open(pdf_image_path)
            st.image(img, caption="PDF debug image", width=True)
            layout_image_path = os.path.join(
                debug_data_path, f"layout_page_{first_page}.png"
            )
            img = Image.open(layout_image_path)
            st.image(img, caption="Layout debug image", width=True)
        st.write("Raw output:")
        st.code(text, language=output_format)
