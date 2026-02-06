import numpy as np
from pydub import AudioSegment
import librosa
from keras.models import load_model
import os

# 加载并预处理音频文件
def preprocess_data(audio_path):
    try:
        raw_audio = AudioSegment.from_file(audio_path)
        samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
        trimmed, _ = librosa.effects.trim(samples, top_db=25)
        padding = max(0, 50000 - len(trimmed))
        padded = np.pad(trimmed, (0, padding), 'constant')
        if padded.shape[0] > 50000:
            return padded[:50000]
        return padded
    except Exception as e:
        print(f"Processing error {audio_path}: {e}")
        return None

# 从音频数据中提取特征
def extract_features(audio):
    zcr_list = []
    rms_list = []
    mfccs_list = []

    FRAME_LENGTH = 2048
    HOP_LENGTH = 512

    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=audio, sr=44100, n_mfcc=13, hop_length=HOP_LENGTH)

    zcr_list.append(zcr)
    rms_list.append(rms)
    mfccs_list.append(mfccs)

    return zcr_list, rms_list, mfccs_list

# 将提取的特征进行拼接
def combine_features(zcr_list, rms_list, mfccs_list):
    zcr_features = np.swapaxes(zcr_list, 1, 2)
    rms_features = np.swapaxes(rms_list, 1, 2)
    mfccs_features = np.swapaxes(mfccs_list, 1, 2)

    X_features = np.concatenate((zcr_features, rms_features, mfccs_features), axis=2)

    return X_features


def predict(audio_path,module_path):
    padded_audio = preprocess_data(audio_path)

    if padded_audio is not None:
        zcr_features, rms_features, mfccs_features = extract_features(padded_audio)

        X_features = combine_features(zcr_features, rms_features, mfccs_features)

        module = load_model(module_path)
        # module.summary()
        prediction = module.predict(X_features)
        print(prediction)
        print('{} 为AI合成的概率为:{:.3f}% 为真的概率为:{:.3f}%'.format(audio_path,prediction[0][0]*100,prediction[0][1]*100))
        predicted_class = np.argmax(prediction, axis=1)

        if predicted_class == 0:
            label = 'fake'
        else:
            label = 'real'

        return label
    else:
        return "Error processing audio file"


# if __name__ == '__main__':
#     result = {}
#     result['real'] = 0
#     result['fake'] = 0
#     new_audio_path = r"F:\dataset_demo\dataset_train\testing\real"
#     # LSTM
#     # model_path = './modules/model_bs32.h5'
#
#     # Transformer
#     model_path = './modules/modelTrans.h5'
#
#     predicted_label = predict(new_audio_path, model_path)
#     print(f"The predicted label for the Audio audio is: {predicted_label}")
#     for audio in os.listdir(new_audio_path):
#         audio_path = os.path.join(new_audio_path, audio)
#         predicted_label = predict(audio_path, model_path)
#         if predicted_label == 'real':
#             result['real'] += 1
#         else:
#             result['fake'] += 1
#         print(f"The predicted label for the Audio audio is: {predicted_label}")
#     print(result)


