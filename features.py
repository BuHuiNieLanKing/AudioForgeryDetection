import librosa
from tqdm import tqdm
import numpy as np
import os

def load_data(data_path):
    """
    加载单个预处理批次的数据
    :param data_path: 批次数据文件路径
    :return: X (音频数据), y (标签)
    """
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    print(f'加载批次 {os.path.basename(data_path)}: X shape={X.shape}, y shape={y.shape}')
    return X, y

def extract_features(X):
    """
    提取单个批次音频的特征（ZCR、RMS、MFCC）
    :param X: 单个批次的音频数据
    :return: zcr_list, rms_list, mfccs_list
    """
    zcr_list = []
    rms_list = []
    mfccs_list = []

    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    for audio in tqdm(X, desc="Extracting Features", unit="file"):
        # 零交叉率
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        # 均方根能量
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        # 梅尔频率倒谱系数
        mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13, hop_length=HOP_LENGTH)

        zcr_list.append(zcr)
        rms_list.append(rms)
        mfccs_list.append(mfccs)

    return zcr_list, rms_list, mfccs_list

def combine_features(zcr_list, rms_list, mfccs_list):
    """
    合并特征并调整维度
    :param zcr_list: 零交叉率特征列表
    :param rms_list: 均方根能量特征列表
    :param mfccs_list: MFCC 特征列表
    :return: 合并后的特征矩阵 (样本数, 时间步, 特征数)
    """
    # 将列表转为 NumPy 数组
    zcr_array = np.array(zcr_list)  # 形状: (样本数, 1, 时间步)
    rms_array = np.array(rms_list)  # 形状: (样本数, 1, 时间步)
    mfccs_array = np.array(mfccs_list)  # 形状: (样本数, 13, 时间步)

    # 调整维度为 (样本数, 时间步, 特征数)
    zcr_reshaped = np.swapaxes(zcr_array, 1, 2)  # 形状: (样本数, 时间步, 1)
    rms_reshaped = np.swapaxes(rms_array, 1, 2)  # 形状: (样本数, 时间步, 1)
    mfccs_reshaped = np.swapaxes(mfccs_array, 1, 2)  # 形状: (样本数, 时间步, 13)

    # 沿特征维度拼接
    X_features = np.concatenate([zcr_reshaped, rms_reshaped, mfccs_reshaped], axis=2)
    return X_features

def save_features(X_features, y, save_dir, batch_id):
    """
    保存特征批次到磁盘
    :param X_features: 特征矩阵
    :param y: 标签
    :param save_dir: 保存目录
    :param batch_id: 批次编号
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'features_batch_{batch_id}.npz')
    np.savez_compressed(save_path, X_features=X_features, y=y)
    # print(f'特征批次 {batch_id} 已保存到 {save_path}')
    s = f'特征批次 {batch_id} 已保存到 {save_path}'
    return s

# if __name__ == '__main__':
#     # 输入输出路径
#     preprocessed_dir = './preprocessed_batches'  # 预处理批次数据目录
#     features_save_dir = './features_batches'     # 特征保存目录
#
#     # 获取所有预处理批次文件
#     preprocessed_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.npz')])
#
#     # 逐个处理批次
#     for batch_file in preprocessed_files:
#         # 加载单个批次数据
#         data_path = os.path.join(preprocessed_dir, batch_file)
#         X, y = load_data(data_path)
#
#         # 提取特征
#         zcr_list, rms_list, mfccs_list = extract_features(X)
#
#         # 合并特征
#         X_features = combine_features(zcr_list, rms_list, mfccs_list)
#
#         # 提取批次编号（假设预处理文件名格式为 preprocess_data_batch_0.npz）
#         batch_id = batch_file.split('_')[-1].split('.')[0]
#
#         # 保存特征批次
#         save_features(X_features, y, features_save_dir, batch_id)
#
#     print('所有批次特征已处理完毕！')