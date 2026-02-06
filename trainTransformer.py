from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Dropout, Dense,GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

def load_features_batches(features_dir):
    """
    加载所有特征批次文件并合并为完整数据集
    :param features_dir: 特征批次文件目录
    :return: X_features (特征数据), y (标签)
    """
    X_features_list = []
    y_list = []

    # 获取所有特征批次文件
    batch_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npz')])

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

# 构建 Transformer 模型
def build_transformer_model(input_shape):
    # 定义输入数据的形状
    inputs = Input(shape=input_shape)

    # 多头自注意力层
    attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output)
    attention_output = Dropout(0.1)(attention_output)

    # 前馈网络
    ff_output = Dense(64, activation='relu')(attention_output)
    ff_output = Dense(64, activation='relu')(ff_output)
    ff_output = LayerNormalization(epsilon=1e-6)(ff_output)
    ff_output = Dropout(0.1)(ff_output)

    # 全局平均池化层和输出层
    outputs = GlobalAveragePooling1D()(ff_output)
    outputs = Dense(2, activation='softmax')(outputs)

    # 返回模型
    modelTrans = Model(inputs=inputs, outputs=outputs)

    return modelTrans

# 模型训练
# def train_model(X_features, y, batch_size):
#     # 数据划分
#     X_train, X_to_split, y_train, y_to_split = train_test_split(X_features, y, test_size=0.3, random_state=1)
#     X_val, X_test, y_val, y_test = train_test_split(X_to_split, y_to_split, test_size=0.4, random_state=1)
#
#     # 标签转换为 one-hot 编码
#     y_train_class = to_categorical(y_train, 2)
#     y_val_class = to_categorical(y_val, 2)
#
#     early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
#     checkpoint = ModelCheckpoint('TransformerModel.keras', monitor='val_loss', verbose=1, save_best_only=True,
#                                  mode='min')
#     rlrop = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.02, patience=100)
#
#     history = modelTrans.fit(X_train, y_train_class, epochs=100, batch_size =  batch_size, validation_data=(X_val, y_val_class),
#                              callbacks=[rlrop, early_stopping, checkpoint])
#
#     return modelTrans, history, X_test, y_test

def train_model(modelTrans, X_features, y, batch_size):
    """
    训练 Transformer 模型
    :param modelTrans: 构建好的Transformer模型
    :param X_features: 特征数据
    :param y: 标签
    :param batch_size: 批次大小
    :return: 训练好的模型、训练历史、测试集数据和标签
    """
    # 数据划分
    X_train, X_to_split, y_train, y_to_split = train_test_split(X_features, y, test_size=0.3, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_to_split, y_to_split, test_size=0.4, random_state=1)

    # 标签转换为 one-hot 编码
    y_train_class = to_categorical(y_train, 2)
    y_val_class = to_categorical(y_val, 2)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
    checkpoint = ModelCheckpoint('TransformerModel.keras', monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')
    rlrop = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.02, patience=100)

    history = modelTrans.fit(X_train, y_train_class, epochs=100, batch_size =  batch_size,
                             validation_data=(X_val, y_val_class), callbacks=[rlrop, early_stopping, checkpoint])

    return modelTrans, history, X_test, y_test

# if __name__ == '__main__':
#     features_dir = './features_batches'
#     X_features, y = load_features_batches(features_dir)
#
#     input_shape = (X_features.shape[1], X_features.shape[2])
#     modelTrans = build_transformer_model(input_shape)
#
#     # 模型编译
#     modelTrans.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_accuracy'])
#
#     modelTrans,history,X_test,y_test = train_model(modelTrans,X_features,y,32)
#
#     test_loss,test_accuracy = modelTrans.evaluate(X_test,to_categorical(y_test))
#     print(f'Test Loss: {test_loss},Test Accuracy: {test_accuracy}')
#
#     modelTrans.save("./modules/modelTrans.h5")  # 保存模型
#     print('模型已经保存在./module/路径下')
