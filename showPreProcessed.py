import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# 设置中文字体（可选，避免中文标签乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def npz_to_waveform_images(npz_file_path, save_root_path, sample_rate=16000):
    """
    从npz文件生成音频波形图片
    :param npz_file_path: 单个.npz文件路径
    :param save_root_path: 图片保存根路径
    :param sample_rate: 音频采样率（默认16000，可根据实际音频调整）
    """
    # 1. 加载npz文件
    data = np.load(npz_file_path)
    X = data['X']  # 音频特征数组，形状：(批次大小, 50000)
    y = data['y']  # 标签数组，形状：(批次大小,) 0=fake，1=real

    # 2. 创建保存目录（按标签分类）
    fake_save_path = os.path.join(save_root_path, 'fake')
    real_save_path = os.path.join(save_root_path, 'real')
    os.makedirs(fake_save_path, exist_ok=True)
    os.makedirs(real_save_path, exist_ok=True)

    # 3. 生成时间轴（对应50000个样本点的时间）
    time = np.linspace(0, len(X[0]) / sample_rate, len(X[0]))

    # 4. 遍历每个音频样本，绘制并保存波形图
    npz_filename = os.path.basename(npz_file_path).replace('.npz', '')  # 获取npz文件名（不含后缀）
    for idx in tqdm(range(len(X)), desc=f"生成{npz_filename}的波形图"):
        audio_data = X[idx]
        label = y[idx]

        # 创建画布
        plt.figure(figsize=(10, 4))  # 宽10英寸，高4英寸
        plt.plot(time, audio_data, color='blue', linewidth=0.5)

        # 设置图表属性
        label_text = '真实音频(real)' if label == 1 else '伪造音频(fake)'
        plt.title(f'音频波形图 - {label_text} (样本{idx})', fontsize=12)
        plt.xlabel('时间 (秒)', fontsize=10)
        plt.ylabel('振幅', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()  # 自动调整布局，避免标签被截断

        # 保存图片（命名规则：npz文件名_样本索引_标签.jpg）
        save_path = real_save_path if label == 1 else fake_save_path
        img_filename = f"{npz_filename}_sample_{idx}_label_{label}.jpg"
        plt.savefig(os.path.join(save_path, img_filename), dpi=100, bbox_inches='tight')
        plt.close()  # 关闭画布，释放内存


def batch_process_npz_files(npz_folder, save_root_path):
    """
    批量处理文件夹下所有npz文件，生成波形图
    :param npz_folder: npz文件所在文件夹
    :param save_root_path: 图片保存根路径
    """
    # 获取所有npz文件
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
    if not npz_files:
        print("未找到任何.npz文件！")
        return

    # 遍历每个npz文件生成图片
    for npz_file in npz_files:
        npz_file_path = os.path.join(npz_folder, npz_file)
        npz_to_waveform_images(npz_file_path, save_root_path)


# if __name__ == '__main__':
#     # 配置路径（根据你的实际路径修改）
#     npz_folder = './preprocessed_batches'  # npz文件所在文件夹
#     save_root_path = './audio_waveform_images'  # 波形图保存根路径
#
#     # 批量生成波形图
#     batch_process_npz_files(npz_folder, save_root_path)
#     print("所有npz文件的波形图已生成完成！")