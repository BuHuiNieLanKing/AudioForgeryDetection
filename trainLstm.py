import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os


def load_features_batches(features_dir):
    """
    加载所有特征批次文件并合并为完整数据集
    :param features_dir: 特征批次文件目录
    :return: X_features (特征数据), y (标签)
    """
    X_features_list = []
    y_list = []

    # 获取并按数字顺序排序特征批次文件
    batch_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npz')],
                         key=lambda x: int(x.split('_')[2].split('.')[0]))

    # 逐个加载批次文件
    for batch_file in batch_files:
        data_path = os.path.join(features_dir, batch_file)
        data = np.load(data_path)
        X_features_list.append(data['X_features'])
        y_list.append(data['y'])

    # 合并所有批次数据
    X_features = np.concatenate(X_features_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    print(f'加载特征数据完成: X_features shape={X_features.shape}, y shape={y.shape}')
    return X_features, y


def train_model(X_features, y, epochs=100, batch_size=32):
    """
    训练 LSTM 模型
    :param X_features: 特征数据
    :param y: 标签
    :param epochs: 训练轮次
    :param batch_size: 批次大小
    :return: 训练好的模型、训练历史、测试集数据和标签
    """
    # 创建保存目录
    os.makedirs('./module', exist_ok=True)

    # 数据集划分（分层抽样）
    X_train, X_to_split, y_train, y_to_split = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_to_split, y_to_split, test_size=0.5, random_state=42, stratify=y_to_split)

    # 将标签转换为 one-hot 编码
    y_train_class = to_categorical(y_train, 2)
    y_val_class = to_categorical(y_val, 2)
    y_test_class = to_categorical(y_test, 2)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
                   input_shape=(X_features.shape[1], X_features.shape[2])))
    model.add(Dropout(0.3))  # 添加Dropout层
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))

    # 模型编译
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSProp',
                  metrics=['categorical_accuracy'])

    # 设置回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                                   verbose=1, mode='min', restore_best_weights=True)
    checkpoint = ModelCheckpoint(
        f'./module/CustomModel_bs{batch_size}.keras',  # 动态路径
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)  # 统一监控val_loss

    # 训练模型
    history = model.fit(
        X_train,y_train_class,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_class),
        callbacks=[early_stopping, checkpoint, rlrop]
    )

    return model, history, X_test, y_test_class


# if __name__ == '__main__':
#     # 特征批次文件目录
#     features_dir = './features_batches/'
#
#     # 加载特征数据
#     X_features, y = load_features_batches(features_dir)
#
#     # 创建模型保存目录
#     os.makedirs('./modules', exist_ok=True)
#
#     # 超参数设置
#     batch_size_list = [8, 16, 32]
#     epochs = 100
#
#     for batch_size in batch_size_list:
#         print(f"\n正在训练 batch_size={batch_size}...")
#         # 训练模型
#         model, history, X_test, y_test = train_model(X_features, y, epochs, batch_size)
#
#         # 评估模型
#         test_loss, test_accuracy = model.evaluate(X_test, y_test)
#         print(f'测试结果 - batch_size={batch_size}: Loss={test_loss:.4f}, Accuracy={test_accuracy:.4f}')
#
#         # 保存最终模型
#         model.save(f"./modules/model_bs{batch_size}.h5")
#         print(f'模型已保存至 ./modules/model_bs{batch_size}.h5')