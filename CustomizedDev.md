# 定制开发


## 环境准备

- [基本开发运行环境安装](SoftwareEngineeringPractices\Programming\ModernSoftwareDev\README.md)
- [大模型相关环境安装](SoftwareEngineeringPractices\Programming\ModernSoftwareDev\Week1\assignment\实验准备和说明.md)



## Debug

```bash
cf marker
streamlit run ./scripts/streamlit_app.py
```

export MAAS_API_KEY=******

## prompts


请翻译为中文，并整理公式以latex的格式，整体以markdown格式进行输出，内容如下:{}


 块级目标函数 (Zhou et al., 2022a). 我们随机遮蔽学生模型接收的输入块，但教师模型不进行遮蔽。随后，我们将学生 iBOT 头应用于学生模型的遮蔽块令牌。同样地，我们对教师模型的 (可见) 块令牌应用教师 iBOT 头，这些块令牌对应于学生模型中被遮蔽的部分。然后按照上述步骤应用 softmax 和中心化操作，得到 iBOT 损失项：

$$ L_{\text{iBOT}} = -\sum_{i} p_{t_i} \log p_{s_i} $$