import pandas as pd
import os
import sys
import numpy as np
import tensorflow as tf
import librosa
from pydub import AudioSegment
from tqdm import tqdm
import datetime

tf.random.set_seed(42)
np.random.seed(42)


def preprocess_data(dataset_path, save_path, batch_size=1000):
    """
    预处理音频数据并分批次保存到磁盘
    :param dataset_path: 数据集路径
    :param save_path: 保存路径
    :param batch_size: 每批次处理的音频数量
    """
    X_batch = []
    y_batch = []

    # 获取当前目录下最大的 batch_count
    batch_count = get_next_batch_count(save_path)

    # 使用 tqdm 包装外层循环，显示进度条
    for variant in tqdm(os.listdir(dataset_path), desc="Processing Variants", unit="variant"):
        variant_path = os.path.join(dataset_path, variant)
        for classesz in tqdm(os.listdir(variant_path), desc="Processing Folder", unit="folder"):
            classes_path = os.path.join(variant_path, classesz)
            for audio in os.listdir(classes_path):
                audio_path = os.path.join(classes_path, audio)
                try:
                    # 加载音频文件
                    raw_audio = AudioSegment.from_file(audio_path)

                    # 将音频数据转换为数值数组
                    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
                    # 去除前后部分的静音，25分贝以下的视作静音
                    trimmed, _ = librosa.effects.trim(samples, top_db=25)
                    # 如果音频的长度不足 50,000，则用零填充
                    padding = max(0, 50000 - len(trimmed))
                    padded = np.pad(trimmed, (0, padding), 'constant')

                    # 取前50000个数据
                    if padded.shape[0] > 50000:
                        padded = padded[:50000]

                    # 将处理后的音频数据（填充后的音频样本）添加到特征列表 X_batch
                    X_batch.append(padded)
                    # 将该音频的类别添加到标签列表 y_batch
                    y_batch.append(classesz)

                    # 如果批次大小达到 batch_size，保存当前批次并清空列表
                    if len(X_batch) >= batch_size:
                        save_batch(X_batch, y_batch, save_path, batch_count)
                        batch_count += 1
                        X_batch.clear()
                        y_batch.clear()

                except Exception as e:
                    print(f"Processing error {audio_path}: {e}")

    # 保存剩余的不足一个批次的数据
    if len(X_batch) > 0:
        save_batch(X_batch, y_batch, save_path, batch_count)


def save_batch(X_batch, y_batch, save_path, batch_count):
    """
    保存当前批次的数据到磁盘
    :param X_batch: 当前批次的特征数据
    :param y_batch: 当前批次的标签数据
    :param save_path: 保存路径
    :param batch_count: 批次编号
    """
    # 将列表转换为数组
    X_batch = np.array(X_batch)
    y_batch = np.array(encode_labels(y_batch))

    # 生成唯一文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    batch_save_path = os.path.join(save_path, f'batch_{batch_count}_{timestamp}.npz')

    # 保存当前批次
    np.savez_compressed(batch_save_path, X=X_batch, y=y_batch)
    print(f'批次 {batch_count} 数据已保存到 {batch_save_path}')


def encode_labels(y):
    """
    将标签编码为 0 和 1
    :param y: 原始标签列表
    :return: 编码后的标签列表
    """
    encoded_labels = []
    for label in y:
        if label == 'real':
            encoded_labels.append(1)
        elif label == 'fake':
            encoded_labels.append(0)
    return encoded_labels


def get_next_batch_count(save_path):
    """
    获取当前保存路径下最大的批次编号，确保批次编号唯一且递增
    :param save_path: 保存路径
    :return: 下一个批次编号
    """
    # 获取当前保存路径下所有 `.npz` 文件
    existing_batches = [f for f in os.listdir(save_path) if f.endswith('.npz')]
    if not existing_batches:
        return 0  # 如果没有现有批次，从 0 开始

    # 提取批次编号并找到最大的批次编号
    batch_numbers = []
    for batch in existing_batches:
        try:
            batch_number = int(batch.split('_')[1])
            batch_numbers.append(batch_number)
        except ValueError:
            continue  # 忽略无法解析的文件

    # 返回下一个批次编号
    return max(batch_numbers) + 1


# if __name__ == '__main__':
#     # 设置数据集路径和保存路径
#     dataset_path_ls = [
#         r'your_dataset_path'
#     ]
#     save_path = './preprocessed_batches'  # 保存批次的文件夹
#     os.makedirs(save_path, exist_ok=True)  # 创建保存路径
#
#     for dataset_path in dataset_path_ls:
#         # 预处理数据并分批次保存
#         preprocess_data(dataset_path, save_path, batch_size=1000)
#
#     print('所有数据已处理并保存完毕')
