# 音频真伪检测系统 / Audio Authenticity Detection System
基于 Flask + TensorFlow 构建的端到端音频真伪检测系统，支持 LSTM/Transformer 双模型训练与推理，覆盖「数据预处理→特征提取→模型训练→可视化分析→批量预测」全流程，可高效识别真实音频与AI生成伪造音频。  
An end-to-end audio authenticity detection system built with Flask + TensorFlow, supporting dual-model (LSTM/Transformer) training and inference. It covers the full workflow of "Data Preprocessing → Feature Extraction → Model Training → Visualization Analysis → Batch Prediction", and can efficiently identify real audio and AI-generated fake audio.

## 📚 目录 / Table of Contents
- [功能特性 (Functional Features)](#-功能特性)
- [环境要求 (Environment Requirements)](#-环境要求)
- [数据集格式要求 (Dataset Format Requirements)](#-数据集格式要求)
- [快速开始 (Quick Start)](#-快速开始)
- [核心模块说明 (Core Module Description)](#-核心模块说明)
- [Web界面使用 (Web Interface Usage)](#-web界面使用)
- [常见问题 (FAQs)](#-常见问题)
- [联系我们 (Contact Us)](#-联系我们)

## 🌟 功能特性 / Functional Features
| 模块 (Module)         | 核心能力 (Core Capabilities)                                                                 |
|-----------------------|----------------------------------------------------------------------------------------------|
| 数据预处理 (Data Preprocessing) | 支持 wav/mp3/flac/m4a/webm 多格式音频；自动标准化采样率/时长；生成结构化NPZ数据 <br> Supports multi-format audio (wav/mp3/flac/m4a/webm); automatically standardizes sampling rate/duration; generates structured NPZ data |
| 特征提取 (Feature Extraction)   | 提取音频核心声学特征：ZCR（过零率）、RMS（均方根能量）、MFCC（梅尔频率倒谱系数） <br> Extracts core acoustic features of audio: ZCR (Zero Crossing Rate), RMS (Root Mean Square Energy), MFCC (Mel-Frequency Cepstral Coefficients) |
| 模型训练 (Model Training)       | 支持 LSTM/Transformer 两种深度学习模型；自定义批次大小、训练轮数、学习率等参数 <br> Supports two deep learning models (LSTM/Transformer); customizes batch size, training epochs, learning rate and other parameters |
| 可视化分析 (Visualization Analysis) | 生成音频波形图、特征热力图（MFCC/差分MFCC）、ZCR/RMS时域曲线；支持结果可视化展示 <br> Generates audio waveform plots, feature heatmaps (MFCC/differential MFCC), ZCR/RMS time-domain curves; supports visual display of results |
| 预测功能 (Prediction Function)   | 单文件预测/文件夹批量预测；实时展示预测进度、置信度及结果分类 <br> Single-file prediction/batch prediction for folders; real-time display of prediction progress, confidence and result classification |
| Web可视化界面 (Web Visual Interface) | 无代码操作全流程；支持任务进度监控、结果一键导出、可视化图表查看 <br> Code-free full-process operation; supports task progress monitoring, one-click result export, and visualization chart viewing |

## 📋 环境要求 / Environment Requirements
### 基础环境 / Basic Environment
- Python 3.8 ~ 3.10（推荐3.9，兼容TensorFlow 2.x，避免3.11+的兼容性问题） <br> Python 3.8 ~ 3.10 (3.9 recommended, compatible with TensorFlow 2.x, avoid compatibility issues with 3.11+)
- 操作系统：Windows 10+/Linux (Ubuntu 18.04+)/macOS 12+ <br> Operating System: Windows 10+/Linux (Ubuntu 18.04+)/macOS 12+
- 硬件：CPU/GPU均可（GPU需配置CUDA 11.2+、cuDNN 8.1+，加速模型训练） <br> Hardware: CPU/GPU are both acceptable (GPU requires CUDA 11.2+ and cuDNN 8.1+ to accelerate model training)

### 依赖工具（必装） / Dependent Tools (Mandatory Installation)
#### 1. FFmpeg（处理非WAV格式音频） / FFmpeg (Process non-WAV format audio)
- Windows：下载 [FFmpeg](https://ffmpeg.org/download.html)，解压后将`bin`目录添加到系统环境变量 <br> Windows: Download [FFmpeg](https://ffmpeg.org/download.html), unzip it and add the `bin` directory to the system environment variables
- Linux：`sudo apt update && sudo apt install ffmpeg`
- macOS：`brew install ffmpeg`

#### 2. Python依赖安装 / Python Dependency Installation
```bash
# 克隆项目 / Clone the project
git clone https://github.com/BuHuiNieLanKing/AudioForgeryDetection.git
cd AudioForgeryDetection

# 安装核心依赖 / Install core dependencies
pip install -r requirements.txt

# 安装dlib（Windows预编译包，Linux/macOS可直接pip install dlib） / Install dlib (precompiled package for Windows; Linux/macOS can directly use pip install dlib)
cd dlib
# 适配Python3.8的预编译包，其他版本需下载对应whl文件 / Precompiled package for Python3.8; download the corresponding whl file for other versions
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl
cd ..  # 回到项目根目录 / Return to the project root directory
```

## 📊 数据集格式要求 / Dataset Format Requirements
### 1. 核心目录结构 / Core Directory Structure
数据集需按「根目录 → 分类子目录」的层级组织，**fake** 目录存放伪造音频，**real** 目录存放真实音频，示例如下： <br>
The dataset must be organized in the hierarchy of "Root Directory → Classification Subdirectories". The **fake** directory stores fake audio, and the **real** directory stores real audio. Example:
```
F:\archive\small_dataset\small_dataset\training  # 数据集根目录（可命名为train/val/testing...） / Dataset root directory (can be named train/val/testing...)
├── fake/                                       # 伪造音频目录（标签0） / Fake audio directory (label 0)
│   ├── fake_audio_1.wav
│   ├── fake_audio_2.mp3
│   └── ...（支持wav/mp3/flac/m4a/webm格式） / ... (supports wav/mp3/flac/m4a/webm formats)
└── real/                                       # 真实音频目录（标签1） / Real audio directory (label 1)
    ├── real_audio_1.wav
    ├── real_audio_2.mp3
    └── ...（支持wav/mp3/flac/m4a/webm格式） / ... (supports wav/mp3/flac/m4a/webm formats)
```

### 2. 格式规范 / Format Specifications
- 目录命名：必须为 `fake`（伪造）和 `real`（真实），区分大小写（建议全小写）； <br> Directory Naming: Must be `fake` (fake) and `real` (real), case-sensitive (all lowercase recommended);
- 音频格式：支持 wav/mp3/flac/m4a/webm，预处理阶段会自动统一格式； <br> Audio Format: Supports wav/mp3/flac/m4a/webm; the format will be automatically unified during preprocessing;
- 路径要求：数据集根目录路径建议为绝对路径（如 `F:\archive\small_dataset\small_dataset\`），避免中文/空格特殊字符； <br> Path Requirements: The root directory path of the dataset is recommended to be an absolute path (e.g., `F:\archive\small_dataset\small_dataset\`), avoiding Chinese/space special characters;
- 数据集划分：建议按 7:2:1 划分训练集（train）、验证集（val）、测试集（testing），各子集均遵循上述目录结构。 <br> Dataset Division: It is recommended to divide into training set (train), validation set (val), and test set (testing) at a ratio of 7:2:1, and each subset follows the above directory structure.

### 3. 适配修改 / Adaptation Modifications
在 `preprocess_data.py` 中修改数据集根目录路径，示例： <br>
Modify the dataset root directory path in `preprocess_data.py`, example:
```python
# 原路径 / Original path
dataset_path = "./data/raw_audio"
# 修改为你的数据集根目录 / Modify to your dataset root directory
dataset_path = r"F:\archive\small_dataset\small_dataset\testing"
```

## 🚀 快速开始 / Quick Start
### 前置准备 / Preparations
1. 确认数据集已按上述格式存放，且路径无中文/空格； <br> Confirm that the dataset is stored in the above format, and the path has no Chinese/space characters;
2. 修改各脚本中的**路径配置**（关键！）： <br> Modify the **path configuration** in each script (critical!):
   - 所有脚本中 `feature_dir`/`audio_dir`/`save_dir` 等路径，替换为你本地路径（如 `F:/AudioForgeryDetection/data/raw_audio`）； <br> Replace paths such as `feature_dir`/`audio_dir`/`save_dir` in all scripts with your local path (e.g., `F:/AudioForgeryDetection/data/raw_audio`);
   - 示例：`feature_dir = './static/features'` → `feature_dir = 'F:/AudioForgeryDetection/static/features'`。 <br> Example: `feature_dir = './static/features'` → `feature_dir = 'F:/AudioForgeryDetection/static/features'`.

### 分步执行（命令行） / Step-by-Step Execution (Command Line)
#### 1. 数据预处理 / Data Preprocessing
```bash
# 标准化音频格式，生成预处理后的NPZ文件 / Standardize audio format and generate preprocessed NPZ files
python preprocess_data.py
```
![1.png](images/1.png)
- 输出：预处理后的音频数据保存至 `./preprocessed_batches` 目录 <br> Output: Preprocessed audio data is saved to the `./preprocessed_batches` directory

#### 2. 特征提取 / Feature Extraction
```bash
# 提取ZCR/RMS/MFCC特征，保存为特征矩阵 / Extract ZCR/RMS/MFCC features and save as feature matrix
python features.py
```
![2.png](images/2.png)
- 输出：特征文件保存至 `./features_batches` 目录（.npz格式，含X_features特征矩阵、y标签） <br> Output: Feature files are saved to the `./features_batches` directory (.npz format, including X_features feature matrix and y labels)

#### 3. 可视化分析 / Visualization Analysis
```bash
# 可视化预处理后的音频波形 / Visualize preprocessed audio waveforms
python showPreProcessed.py

# 可视化特征分布（MFCC热力图、ZCR/RMS曲线、MFCC差分图） / Visualize feature distribution (MFCC heatmap, ZCR/RMS curve, MFCC differential map)
python showFeatures.py
```
![3.png](images/3.png)
![5.jpg](images/5.jpg)

![4.png](images/4.png)

![6.png](images/6.png)

![7.png](images/7.png)

![8.png](images/8.png)
- 输出：可视化图片保存至 `./audio_waveform_images,./feature_visualizations` 目录，按「real/fake」分类存储 <br> Output: Visualized images are saved to `./audio_waveform_images,./feature_visualizations` directories, stored by "real/fake" classification

#### 4. 模型训练 / Model Training
```bash
# 训练LSTM模型 / Train LSTM model
python trainLstm.py
# 训练Transformer模型 / Train Transformer model
python trainTransformer.py
```
![9.png](images/9.png)

![16.png](images/16.png)
- 输出：训练好的模型保存至 `models` 目录（.h5格式） <br> Output: Trained models are saved to the `models` directory (.h5 format)

#### 5. 批量预测 / Batch Prediction
```bash
# 对目标文件夹内的音频进行真伪检测 / Perform authenticity detection on audio in the target folder
python predict.py
```

Lstm 模型预测结果 / LSTM Model Prediction Results
![10.png](images/10.png)
![11.png](images/11.png)

Transformer 模型预测结果 / Transformer Model Prediction Results
![12.png](images/12.png)
![13.png](images/13.png)

- 输入：需预测的音频文件夹路径（在predict.py中配置）； <br> Input: Path of the audio folder to be predicted (configured in predict.py);
- 输出：含文件名、预测标签、置信度。 <br> Output: Includes file name, predicted label, and confidence.

## 🖥️ Web界面使用 / Web Interface Usage
### 启动Web服务 / Start Web Service
```bash
# 运行Flask Web应用 / Run Flask Web application
python app.py
```
![14.png](images/14.png)

![15.png](images/15.png)
### 操作流程 / Operation Process
1. 浏览器访问 `http://localhost:5000`（默认端口）； <br> Access `http://localhost:5000` (default port) in the browser;
2. 左侧导航栏依次执行： <br> Execute sequentially from the left navigation bar:
   - 「数据上传」：上传待处理的音频文件； <br> "Data Upload": Upload audio files to be processed;
   - 「数据预处理」：点击开始按钮，等待预处理完成； <br> "Data Preprocessing": Click the start button and wait for preprocessing to complete;
   - 「特征提取」：自动提取声学特征； <br> "Feature Extraction": Automatically extract acoustic features;
   - 「模型训练」：选择LSTM/Transformer，配置参数后开始训练； <br> "Model Training": Select LSTM/Transformer, configure parameters and start training;
   - 「可视化分析」：查看特征热力图、波形图等； <br> "Visualization Analysis": View feature heatmaps, waveform plots, etc.;
   - 「预测」：上传音频文件，实时查看预测结果。 <br> "Prediction": Upload audio files and view prediction results in real time.

## 📖 核心模块说明 / Core Module Description
| 脚本文件 (Script Files)   | 核心作用 (Core Function)                                                                 |
|---------------------------|-----------------------------------------------------------------------------------------|
| preprocess_data.py        | 音频格式标准化、时长裁剪、采样率统一，生成模型可读取的NPZ数据 <br> Standardizes audio format, cuts duration, unifies sampling rate, and generates NPZ data readable by the model |
| features.py               | 提取ZCR/RMS/MFCC特征，拼接为15维特征矩阵，按批次保存为.npz文件 <br> Extracts ZCR/RMS/MFCC features, splices into a 15-dimensional feature matrix, and saves as .npz files in batches |
| showPreProcessed.py       | 可视化预处理后的音频波形，直观查看音频时域特征 <br> Visualizes preprocessed audio waveforms to intuitively view audio time-domain features |
| showFeatures.py           | 可视化声学特征：MFCC热力图、ZCR/RMS时域曲线、MFCC一阶/二阶差分图 <br> Visualizes acoustic features: MFCC heatmap, ZCR/RMS time-domain curve, MFCC first/second-order differential map |
| trainLstm.py              | 基于LSTM网络训练音频真伪分类模型，适合序列特征建模 <br> Trains audio authenticity classification model based on LSTM network, suitable for sequence feature modeling |
| trainTransformer.py       | 基于Transformer的自注意力机制训练模型，捕捉长距离特征依赖 <br> Trains model based on Transformer's self-attention mechanism to capture long-distance feature dependencies |
| predict.py                | 加载训练好的模型，支持单文件/批量音频预测，输出分类结果和置信度 <br> Loads trained models, supports single-file/batch audio prediction, and outputs classification results and confidence |
| app.py                    | Flask Web服务入口，封装所有功能为可视化界面 <br> Flask Web service entry, encapsulates all functions into a visual interface |


## 📞 联系我们 / Contact Us
- QQ：310720949
- 问题反馈：可在GitHub项目下提交Issue，或通过QQ反馈使用过程中的问题 <br> Feedback: You can submit an Issue under the GitHub project, or feedback problems during use via QQ

## 📄 许可证 / License
本项目仅供学习和研究使用，禁止用于商业用途。 <br> This project is for learning and research purposes only, and is prohibited for commercial use.

### 总结 / Summary
1. 文档已全面适配中英双语，保持原有结构、代码块和图片引用不变，仅对描述性文字补充英文翻译； <br> The document has been fully adapted to Chinese and English bilingual, keeping the original structure, code blocks and image references unchanged, only adding English translations to descriptive text.
2. 技术术语采用行业通用英文表达（如MFCC、LSTM、Transformer），确保专业准确性； <br> Technical terms use industry-general English expressions (e.g., MFCC, LSTM, Transformer) to ensure professional accuracy.
3. 保留所有操作指令和路径配置示例，双语对照便于国内外用户理解和使用。 <br> All operation instructions and path configuration examples are retained, with bilingual comparison to facilitate understanding and use by users at home and abroad.