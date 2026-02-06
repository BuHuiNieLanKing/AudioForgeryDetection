import numpy as np
import matplotlib

# 1. 核心修复：设置非交互式后端，彻底禁用GUI，消除线程警告
matplotlib.use('Agg')  # 仅保存图片，不启动GUI，解决子线程警告
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties
from tqdm import tqdm


# ===================== 修复字体设置 + 中文显示 =====================
def set_chinese_font_global():
    """全局设置中文字体（修复seaborn/colorbar字体警告）"""
    # 全局字体配置（所有plt/seaborn都会继承）
    font_paths = [
        'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/msyhbd.ttc',  # 微软雅黑粗体
    ]
    valid_font_path = None
    for path in font_paths:
        if os.path.exists(path):
            valid_font_path = path
            break
    if valid_font_path:
        # 设置全局字体
        plt.rcParams['font.family'] = 'Microsoft YaHei'  # 对应msyh.ttc
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        print(f"成功加载全局中文字体：{valid_font_path}")
    else:
        print("警告：未找到中文字体文件，中文可能显示为方框！")

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    sns.set_style("whitegrid")
    # 修复seaborn字体
    sns.set(font='Microsoft YaHei')


# 新增：MFCC差分计算函数（核心修复mfcc_delta无数据问题）
def compute_mfcc_delta(mfcc, order=1):
    """
    计算MFCC的一阶/二阶差分
    :param mfcc: MFCC特征，shape=(帧数, 13)
    :param order: 差分阶数（1=一阶，2=二阶）
    :return: 差分特征，shape=(帧数, 13)（与原MFCC维度一致）
    """
    if mfcc.ndim != 2 or mfcc.shape[1] != 13:
        print(f"警告：MFCC维度异常（{mfcc.shape}），预期(帧数, 13)！")
        return np.zeros_like(mfcc)

    # 一阶差分：后项减前项，补0保持维度一致
    if order == 1:
        delta = np.diff(mfcc, n=1, axis=0)
        delta = np.pad(delta, ((0, 1), (0, 0)), mode='constant')  # 补最后一行0
    # 二阶差分：对一阶差分再做差分
    elif order == 2:
        delta1 = np.diff(mfcc, n=1, axis=0)
        delta1 = np.pad(delta1, ((0, 1), (0, 0)), mode='constant')
        delta2 = np.diff(delta1, n=1, axis=0)
        delta2 = np.pad(delta2, ((0, 1), (0, 0)), mode='constant')
        delta = delta2
    else:
        delta = mfcc  # 默认返回原MFCC

    return delta


# 执行全局字体设置
set_chinese_font_global()


def load_feature_data(feature_path):
    """加载单个特征批次文件"""
    # 新增：基础路径校验（保留原有逻辑，仅增强鲁棒性）
    if not os.path.exists(feature_path):
        print(f"错误：特征文件不存在 → {feature_path}")
        return None, None
    try:
        data = np.load(feature_path)
        X_features = data['X_features']  # (样本数, 98, 15) 修正维度注释
        y = data['y']  # (样本数,) 0=fake/1=real
        print(f'加载特征批次 {os.path.basename(feature_path)}: X_features shape={X_features.shape}, y shape={y.shape}')
        return X_features, y
    except Exception as e:
        print(f"加载特征文件失败：{e}")
        return None, None


def split_features(combined_features):
    """
    2. 核心修复：匹配实际特征维度 (98帧 × 15维)
    原代码按41维拆分，现按15维拆分（根据你的X_features shape=(120,98,15)）
    新增：计算MFCC差分，不再返回None
    """
    # 新增：维度校验（保留原有拆分逻辑）
    if combined_features.ndim != 2 or combined_features.shape[1] != 15:
        print(f"警告：特征维度异常（{combined_features.shape}），预期(98, 15)！")
        return None, None, None, None, None

    # 假设15维特征拼接顺序：ZCR(1) + RMS(1) + MFCC(13)
    zcr = combined_features[:, 0:1]  # 98帧 × 1维（ZCR）
    rms = combined_features[:, 1:2]  # 98帧 × 1维（RMS）
    mfcc = combined_features[:, 2:15]  # 98帧 × 13维（MFCC）

    # 核心修改：计算差分（不再返回None）
    delta1 = compute_mfcc_delta(mfcc, order=1)  # 一阶差分
    delta2 = compute_mfcc_delta(mfcc, order=2)  # 二阶差分

    return zcr, rms, mfcc, delta1, delta2


def create_save_dirs(save_root_dir, label, batch_name):
    """创建分层存储目录"""
    label_dir = 'real' if label == 1 else 'fake'
    dir_config = {
        'mfcc': os.path.join(save_root_dir, label_dir, 'mfcc'),
        'zcr_rms': os.path.join(save_root_dir, label_dir, 'zcr_rms'),
        'mfcc_delta': os.path.join(save_root_dir, label_dir, 'mfcc_delta')
    }
    for dir_path in dir_config.values():
        os.makedirs(dir_path, exist_ok=True)
    return dir_config, label_dir


def visualize_single_sample(sample_idx, features, label, save_root_dir, batch_name):
    """可视化单个样本（移除局部字体设置，使用全局配置）"""
    # ========== 核心修复：修改None判断逻辑 ==========
    zcr, rms, mfcc, delta1, delta2 = split_features(features)
    # 正确判断：检查是否有任意一个值为None（而非用None in 数组元组）
    if zcr is None or rms is None or mfcc is None:
        print(f"跳过样本{sample_idx}：特征拆分失败")
        return
    # ===============================================

    label_text = '真实音频(real)' if label == 1 else '伪造音频(fake)'
    dir_config, label_dir = create_save_dirs(save_root_dir, label, batch_name)
    # 修正时间帧数量（98帧，原100帧）
    time_frames = np.arange(0, len(zcr))  # 动态适配帧数量，避免硬编码

    # 1. MFCC热力图（使用全局字体，移除局部fontproperties）
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # 无需局部字体设置，全局已生效
    heatmap = sns.heatmap(mfcc.T, cmap='viridis',
                          xticklabels=10, yticklabels=2,
                          cbar_kws={'label': 'MFCC系数值'},
                          ax=ax)
    ax.set_aspect('auto')
    # 标题/坐标轴直接使用全局字体
    plt.title(f'{label_text} - MFCC热力图（样本{sample_idx}）', fontsize=12)
    plt.xlabel('时间帧（共{}帧）'.format(len(time_frames)), fontsize=10)
    plt.ylabel('MFCC维度（1-13）', fontsize=10)
    plt.tight_layout()
    # 保存图片
    mfcc_save_path = os.path.join(dir_config['mfcc'], f'{batch_name}_sample{sample_idx}_mfcc.png')
    plt.savefig(mfcc_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 2. ZCR+RMS时域曲线（适配98帧）
    plt.figure(figsize=(12, 4))
    plt.plot(time_frames, zcr[:, 0], label='过零率(ZCR)', color='blue', linewidth=1.5)
    plt.plot(time_frames, rms[:, 0], label='均方根能量(RMS)', color='red', linewidth=1.5)
    plt.title(f'{label_text} - ZCR+RMS时域曲线（样本{sample_idx}）', fontsize=12)
    plt.xlabel('时间帧', fontsize=10)
    plt.ylabel('特征值', fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()
    zcr_rms_save_path = os.path.join(dir_config['zcr_rms'], f'{batch_name}_sample{sample_idx}_zcr_rms.png')
    plt.savefig(zcr_rms_save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 3. MFCC差分对比图（核心修复：现在delta1/delta2有数据，会执行该部分）
    if delta1 is not None and delta2 is not None:
        # 子图1：一阶差分热力图
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        sns.heatmap(delta1.T, cmap='plasma',
                    xticklabels=10, yticklabels=2,
                    cbar_kws={'label': '一阶差分系数值'},
                    ax=ax1)
        ax1.set_aspect('auto')
        plt.title(f'{label_text} - MFCC一阶差分热力图（样本{sample_idx}）', fontsize=12)
        plt.xlabel('时间帧（共{}帧）'.format(len(time_frames)), fontsize=10)
        plt.ylabel('MFCC维度（1-13）', fontsize=10)
        plt.tight_layout()
        delta1_save_path = os.path.join(dir_config['mfcc_delta'], f'{batch_name}_sample{sample_idx}_mfcc_delta1.png')
        plt.savefig(delta1_save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 子图2：二阶差分热力图
        plt.figure(figsize=(12, 6))
        ax2 = plt.gca()
        sns.heatmap(delta2.T, cmap='inferno',
                    xticklabels=10, yticklabels=2,
                    cbar_kws={'label': '二阶差分系数值'},
                    ax=ax2)
        ax2.set_aspect('auto')
        plt.title(f'{label_text} - MFCC二阶差分热力图（样本{sample_idx}）', fontsize=12)
        plt.xlabel('时间帧（共{}帧）'.format(len(time_frames)), fontsize=10)
        plt.ylabel('MFCC维度（1-13）', fontsize=10)
        plt.tight_layout()
        delta2_save_path = os.path.join(dir_config['mfcc_delta'], f'{batch_name}_sample{sample_idx}_mfcc_delta2.png')
        plt.savefig(delta2_save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 额外：差分对比曲线（保留原有对比逻辑，可选）
        plt.figure(figsize=(15, 5))
        mfcc_dim = 5  # 选第6个MFCC维度做对比
        # 子图1：原始MFCC
        plt.subplot(1, 3, 1)
        plt.plot(time_frames, mfcc[:, mfcc_dim], color='blue', label='MFCC')
        plt.title(f'MFCC第{mfcc_dim + 1}维', fontsize=10)
        plt.xlabel('时间帧')
        plt.ylabel('特征值')
        plt.legend()
        # 子图2：一阶差分
        plt.subplot(1, 3, 2)
        plt.plot(time_frames, delta1[:, mfcc_dim], color='orange', label='一阶差分')
        plt.title(f'一阶差分第{mfcc_dim + 1}维', fontsize=10)
        plt.xlabel('时间帧')
        plt.ylabel('特征值')
        plt.legend()
        # 子图3：二阶差分
        plt.subplot(1, 3, 3)
        plt.plot(time_frames, delta2[:, mfcc_dim], color='green', label='二阶差分')
        plt.title(f'二阶差分第{mfcc_dim + 1}维', fontsize=10)
        plt.xlabel('时间帧')
        plt.ylabel('特征值')
        plt.legend()
        # 总标题
        plt.suptitle(f'{label_text} - MFCC差分对比（样本{sample_idx}）', fontsize=12)
        plt.tight_layout()
        delta_compare_save_path = os.path.join(dir_config['mfcc_delta'],
                                               f'{batch_name}_sample{sample_idx}_mfcc_delta_compare.png')
        plt.savefig(delta_compare_save_path, dpi=150, bbox_inches='tight')
        plt.close()


def visualize_feature_batch(feature_path, save_root_dir):
    """可视化单个批次的全部样本（优化tqdm子线程显示）"""
    # 新增：空值校验（保留原有逻辑）
    X_features, y = load_feature_data(feature_path)
    if X_features is None or y is None:
        print(f"跳过批次：{os.path.basename(feature_path)} 加载失败")
        return

    batch_name = os.path.basename(feature_path).replace('.npz', '')
    sample_count = len(X_features)
    print(f'可视化该批次全部{sample_count}个样本...')

    # 3. 修复子线程中tqdm进度条显示
    for sample_idx in tqdm(range(sample_count),
                           desc=f'可视化{batch_name}',
                           leave=True,  # 保留进度条
                           ncols=80):  # 适配终端宽度
        sample_features = X_features[sample_idx]
        sample_label = y[sample_idx]
        visualize_single_sample(sample_idx, sample_features, sample_label, save_root_dir, batch_name)


def batch_visualize_features(feature_dir, save_root_dir):
    """批量可视化所有批次"""
    # 新增：目录校验（保留原有逻辑）
    if not os.path.exists(feature_dir):
        print(f"错误：特征目录不存在 → {feature_dir}")
        return

    feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('.npz')])
    if not feature_files:
        print("未找到任何特征文件！")
        return

    for feature_file in feature_files:
        feature_path = os.path.join(feature_dir, feature_file)
        visualize_feature_batch(feature_path, save_root_dir)


# if __name__ == '__main__':
#     feature_dir = './static/features'
#     visualize_save_dir = './feature_visualizations'
#     # 新增：创建保存目录（避免首次运行报错）
#     os.makedirs(visualize_save_dir, exist_ok=True)
#     batch_visualize_features(feature_dir, visualize_save_dir)
#     print(f'特征可视化完成！图片保存在 {visualize_save_dir}')
#     print(f'存储结构：{visualize_save_dir}/[real/fake]/[mfcc/zcr_rms/mfcc_delta]/')