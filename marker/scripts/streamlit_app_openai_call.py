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


<<<<<<< Updated upstream
def markdown_insert_images(markdown, images):
=======
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

def img2cloud_to_html(img, filename, img_alt):
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
    unique_filename = f"screenshots/{filename}-{uuid.uuid4().hex}.{file_ext}"

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
def markdown_insert_images(markdown, filename, images):
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
                image_markdown, img_to_html(images[image_path], image_alt)
=======
                image_markdown, img2cloud_to_html(images[image_path], filename, image_alt)
>>>>>>> Stashed changes
            )
    return markdown

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
                {"role": "system", "content": f"请提供翻译的原材料内容,系统会将其翻译为中文并做一定的整理工作，如将文中的'我们'用'论文中'进行替换,将公式进行整理成latex的格式表示,整体保留markdown格式进行输出等。"},
                {"role": "user", "content": f"好的，内容如下:{text}。请帮助翻译和整理，谢谢。"}
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
col1, col2 = st.columns([0.5, 0.5])

model_dict = load_models()
cli_options = parse_args()

st.markdown("""
# Marker Demo

This app will let you try marker, a PDF or image -> Markdown, HTML, JSON converter. It works with any language, and extracts images, tables, equations, etc.

Find the project [here](https://github.com/VikParuchuri/marker).
""")

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

with col1:
    page_count = page_count(in_file)
    page_number = st.number_input(
        f"Page number out of {page_count}:", min_value=0, value=0, max_value=page_count
    )
    pil_image = get_page_image(in_file, page_number)
    st.image(pil_image, use_container_width=True)

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

text, ext, images = text_from_rendered(rendered)
with col2:
    if output_format == "markdown":
        filename = os.path.basename(in_file)
        text = markdown_insert_images(text, filename, images)
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
            st.markdown(text, unsafe_allow_html=True)
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
            st.image(img, caption="PDF debug image", use_container_width=True)
            layout_image_path = os.path.join(
                debug_data_path, f"layout_page_{first_page}.png"
            )
            img = Image.open(layout_image_path)
            st.image(img, caption="Layout debug image", use_container_width=True)
        st.write("Raw output:")
        st.code(text, language=output_format)
