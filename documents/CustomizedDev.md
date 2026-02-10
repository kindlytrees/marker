# 定制开发

## 环境准备
- [基本开发运行环境安装](SoftwareEngineeringPractices\Programming\ModernSoftwareDev\README.md)
- [大模型相关环境安装](SoftwareEngineeringPractices\Programming\ModernSoftwareDev\Week1\assignment\实验准备和说明.md)


## 本地部署运行

```bash
cf marker
pip install oss2
pip install PyMuPDF
pip install --upgrade streamlit
export MAAS_API_KEY=******
export *****

streamlit run ./scripts/streamlit_app_openai_call.py
```
## 现有代码实现的理解记录

rendered = convert_pdf(temp_pdf, config_parser)

`convert_pdf`实现了将pdf转换为markdown时，rendered有markdown的问题和返回的图像属性（如路径等，不是图像的内存表示）
if isinstance(rendered, MarkdownOutput):
    return rendered.markdown, "md", rendered.images

因此需要进一步追根溯源，找到保存图像的地方，并调用云存储api，并返回url
```python
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
```
类`PdfConverter`的实现路径为：`marker\marker\converters\pdf.py`
类`DocumentBuilder`的实现路径为：`marker\marker\builders\document.py`

LayoutBuilder通过surya_layout对文档的每一页（以图像形式进行布局检测，返回blocks（以多边形定义））

```python
#marker\marker\renderers\html.py
def extract_image(self, document, image_id):
    image_block = document.get_block(image_id)
    cropped = image_block.get_image(
        document, highres=self.image_extraction_mode == "highres"
    )
    return cropped
```

```python
#images[image_path]返回的并不是路径，而是iostream，因此，只需要对img_to_html函数进行修改就可以
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
                image_markdown, img_to_html(images[image_path], image_alt)
            )
    return markdown
```

## prompts

```json
    messages=[
        {"role": "system", "content": f"请提供翻译的原材料内容,系统会将其翻译为中文并进行一定的整理，需求如下： \
            1、图像标签的内容保持不变; \
            2、将翻译后文中出现的'我们'用'论文中'进行替换； \
            3、将公式以latex的格式进行整理; \
            4、整体结果以markdown格式输出。"},
        {"role": "user", "content": f"好的，内容如下:{text}。请帮助翻译整理和修订，谢谢。"}
    ]
```

## TODO
- [x] 去除转换中的base64编码的图像数据再交给翻译
- [ ] 做成一个服务发布出来，提供上传pdf，并指定处理的页面范围，直接和已有的doos/md应用集成