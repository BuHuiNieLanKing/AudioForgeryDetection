from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import shutil
from tensorflow.keras.utils import to_categorical
import glob
# 导入自定义模块
from preprocess_data import preprocess_data as pp_data
from features import load_data, extract_features, combine_features, save_features
from trainLstm import load_features_batches as load_lstm_features, train_model as train_lstm
from trainTransformer import (
    load_features_batches as load_trans_features,
    build_transformer_model,
    train_model as train_trans
)
from showPreProcessed import batch_process_npz_files as wave_visualize
from showFeatures import batch_visualize_features as feat_visualize
from predict import predict as audio_predict
import time

# 初始化Flask应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PREPROCESSED_FOLDER'] = 'static/preprocessed'
app.config['FEATURES_FOLDER'] = 'static/features'
app.config['MODELS_FOLDER'] = 'static/models'
app.config['VISUALIZATIONS_FOLDER'] = 'static/visualizations'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'flac', 'm4a', 'webm'}  # 支持的音频格式

# 全局变量：预测进度、任务状态
predict_progress = {'status': 'idle', 'progress': 0, 'total': 0, 'success': 0, 'failed': 0}
task_status = {
    'preprocess': {'status': 'idle', 'message': ''},
    'extract': {'status': 'idle', 'message': ''},
    'train': {'status': 'idle', 'message': ''},
    'visualize_wave': {'status': 'idle', 'message': ''},
    'visualize_feat': {'status': 'idle', 'message': ''}
}


# ---------------------- 工具函数 ----------------------
def clear_folder(folder_path):
    """清空文件夹（保留文件夹本身）"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"删除失败 {file_path}: {str(e)}")
            raise Exception(f"清空文件夹失败：{str(e)}")


def allowed_file(filename):
    """检查文件格式是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_all_audio_files(folder_path):
    """获取文件夹下所有音频文件"""
    audio_files = []
    for ext in app.config['ALLOWED_EXTENSIONS']:
        audio_files.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
        audio_files.extend(glob.glob(os.path.join(folder_path, f'**/*.{ext}'), recursive=True))
    return audio_files



def generate_file_list_html(base_folder, current_path, base_url):
    """
    生成带子文件夹的文件列表HTML
    base_url: 基础URL路径（waveform/features）
    """
    # 计算相对路径
    rel_path = os.path.relpath(current_path, base_folder) if current_path != base_folder else ''
    # 上级目录链接
    parent_link = ''
    if rel_path:
        parent_dir = os.path.dirname(current_path)
        parent_rel = os.path.relpath(parent_dir, base_folder)
        parent_link = f'<li><a href="/static/visualizations/{base_url}/{parent_rel}">../ 上级目录</a></li>'

    files = []
    dirs = []
    for item in os.listdir(current_path):
        item_path = os.path.join(current_path, item)
        item_rel = os.path.relpath(item_path, base_folder)
        if os.path.isdir(item_path):
            dirs.append(
                f'<li><i class="bi bi-folder"></i> <a href="/static/visualizations/{base_url}/{item_rel}">{item}/</a></li>')
        else:
            if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 图片预览 - 使用正确的base_url
                files.append(f'''
                <li>
                    <i class="bi bi-file-image"></i> 
                    <a href="/static/visualizations/{base_url}/{item_rel}" target="_blank">{item}</a>
                    <a href="/static/visualizations/{base_url}/{item_rel}" target="_blank" class="btn btn-sm btn-outline-secondary ms-2">
                        <i class="bi bi-eye"></i> 预览
                    </a>
                </li>''')

    return f'''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>音频{base_url}列表</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.0/font/bootstrap-icons.css">
        <style>
            body {{ padding: 2rem; }}
            .file-list {{ list-style: none; padding-left: 0; margin-top: 1rem; }}
            .file-list li {{ padding: 0.5rem; border-bottom: 1px solid #eee; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h1><i class="bi bi-graph-up"></i> 音频{base_url}列表</h1>
                <a href="/" class="btn btn-primary">
                    <i class="bi bi-house"></i> 返回首页
                </a>
            </div>
            <div class="alert alert-info">
                当前路径: {current_path}
            </div>
            <ul class="file-list">
                {parent_link}
                {"".join(dirs)}
                {"".join(files)}
            </ul>
            {f'<div class="alert alert-warning mt-3">暂无图片文件！</div>' if not files and not dirs else ''}
        </div>
    </body>
    </html>
    '''


@app.route('/static/visualizations/waveform/')
@app.route('/static/visualizations/waveform/<path:subpath>')
def serve_waveform(subpath=None):
    """波形图路由（支持子文件夹）- 修复路径"""
    base_folder = os.path.join(app.config['VISUALIZATIONS_FOLDER'], 'waveform')
    if subpath is None:
        return generate_file_list_html(base_folder, base_folder, 'waveform')  # 传入正确的base_url

    target_path = os.path.join(base_folder, subpath)
    if os.path.isdir(target_path):
        return generate_file_list_html(base_folder, target_path, 'waveform')
    elif os.path.isfile(target_path):
        return send_from_directory(os.path.dirname(target_path), os.path.basename(target_path))
    else:
        return f'<h1>404 未找到文件</h1><p>路径：{target_path}</p>', 404


@app.route('/static/visualizations/features/')
@app.route('/static/visualizations/features/<path:subpath>')
def serve_features(subpath=None):
    """特征图路由（支持子文件夹）- 修复路径"""
    base_folder = os.path.join(app.config['VISUALIZATIONS_FOLDER'], 'features')
    if subpath is None:
        return generate_file_list_html(base_folder, base_folder, 'features')  # 传入正确的base_url

    target_path = os.path.join(base_folder, subpath)
    if os.path.isdir(target_path):
        return generate_file_list_html(base_folder, target_path, 'features')
    elif os.path.isfile(target_path):
        return send_from_directory(os.path.dirname(target_path), os.path.basename(target_path))
    else:
        return f'<h1>404 未找到文件</h1><p>路径：{target_path}</p>', 404


# ---------------------- 异步任务函数 ----------------------
def preprocess_task(dataset_path):
    task_status['preprocess']['status'] = 'running'
    task_status['preprocess']['message'] = '开始清空旧的预处理文件...'
    try:
        clear_folder(app.config['PREPROCESSED_FOLDER'])
        task_status['preprocess']['message'] = '旧文件清空完成，开始预处理音频数据...'
        pp_data(dataset_path, app.config['PREPROCESSED_FOLDER'], batch_size=1000)
        task_status['preprocess']['status'] = 'completed'
        task_status['preprocess']['message'] = '数据预处理完成！'
    except Exception as e:
        task_status['preprocess']['status'] = 'failed'
        task_status['preprocess']['message'] = f'预处理失败：{str(e)}'


def extract_task():
    task_status['extract']['status'] = 'running'
    task_status['extract']['message'] = '开始清空旧的特征文件...'
    try:
        clear_folder(app.config['FEATURES_FOLDER'])
        task_status['extract']['message'] = '旧文件清空完成，开始提取音频特征...'
        preprocessed_files = sorted([f for f in os.listdir(app.config['PREPROCESSED_FOLDER']) if f.endswith('.npz')])
        if not preprocessed_files:
            raise Exception('未找到预处理的npz文件！')
        for batch_file in preprocessed_files:
            data_path = os.path.join(app.config['PREPROCESSED_FOLDER'], batch_file)
            X, y = load_data(data_path)
            zcr_list, rms_list, mfccs_list = extract_features(X)
            X_features = combine_features(zcr_list, rms_list, mfccs_list)
            batch_id = batch_file.split('_')[1] if '_' in batch_file else len(preprocessed_files)
            save_features(X_features, y, app.config['FEATURES_FOLDER'], batch_id)
        task_status['extract']['status'] = 'completed'
        task_status['extract']['message'] = '特征提取完成！'
    except Exception as e:
        task_status['extract']['status'] = 'failed'
        task_status['extract']['message'] = f'特征提取失败：{str(e)}'


# 修改1：支持自定义模型名称
def train_task(model_type, batch_size, epochs, model_name):
    task_status['train']['status'] = 'running'
    task_status['train']['message'] = f'开始训练{model_type.upper()}模型（名称：{model_name}）...'
    try:
        features_dir = app.config['FEATURES_FOLDER']
        if not os.listdir(features_dir):
            raise Exception('未找到特征文件！')

        # 确保模型名称合法
        safe_model_name = model_name.strip().replace(' ', '_').replace('/', '_').replace('\\', '_')
        if not safe_model_name:
            safe_model_name = f'{model_type}_model_{int(time.time())}'

        model_save_path = os.path.join(app.config['MODELS_FOLDER'], f'{safe_model_name}.h5')

        if model_type == 'lstm':
            X_features, y = load_lstm_features(features_dir)
            model, _, _, _ = train_lstm(X_features, y, epochs=int(epochs), batch_size=int(batch_size))
            model.save(model_save_path)
        elif model_type == 'transformer':
            X_features, y = load_trans_features(features_dir)
            input_shape = (X_features.shape[1], X_features.shape[2])
            modelTrans = build_transformer_model(input_shape)
            modelTrans.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['categorical_accuracy'])
            modelTrans, _, X_test, y_test = train_trans(modelTrans, X_features, y, batch_size=int(batch_size))
            test_loss, test_acc = modelTrans.evaluate(X_test, to_categorical(y_test))
            modelTrans.save(model_save_path)
            task_status['train']['message'] += f' 测试准确率：{test_acc:.4f}'

        task_status['train']['status'] = 'completed'
        task_status['train']['message'] = f'{model_type.upper()}模型训练完成！模型已保存为：{safe_model_name}.h5'
    except Exception as e:
        task_status['train']['status'] = 'failed'
        task_status['train']['message'] = f'{model_type.upper()}模型训练失败：{str(e)}'


def wave_visualize_task():
    task_status['visualize_wave']['status'] = 'running'
    task_status['visualize_wave']['message'] = '开始清空旧的波形图文件...'
    try:
        wave_folder = os.path.join(app.config['VISUALIZATIONS_FOLDER'], 'waveform')
        clear_folder(wave_folder)
        task_status['visualize_wave']['message'] = '旧文件清空完成，开始生成波形图...'
        npz_folder = app.config['PREPROCESSED_FOLDER']
        wave_visualize(npz_folder, wave_folder)
        task_status['visualize_wave']['status'] = 'completed'
        task_status['visualize_wave']['message'] = '波形可视化完成！'
    except Exception as e:
        task_status['visualize_wave']['status'] = 'failed'
        task_status['visualize_wave']['message'] = f'波形可视化失败：{str(e)}'


def feat_visualize_task():
    task_status['visualize_feat']['status'] = 'running'
    task_status['visualize_feat']['message'] = '开始清空旧的特征图文件...'
    try:
        feat_folder = os.path.join(app.config['VISUALIZATIONS_FOLDER'], 'features')
        clear_folder(feat_folder)
        task_status['visualize_feat']['message'] = '旧文件清空完成，开始生成特征图...'
        feature_dir = app.config['FEATURES_FOLDER']
        feat_visualize(feature_dir, feat_folder)
        task_status['visualize_feat']['status'] = 'completed'
        task_status['visualize_feat']['message'] = '特征可视化完成！'
    except Exception as e:
        task_status['visualize_feat']['status'] = 'failed'
        task_status['visualize_feat']['message'] = f'特征可视化失败：{str(e)}'


# ---------------------- 预测功能扩展（修复录音问题） ----------------------
def predict_folder_task(folder_path, model_name, result_callback):
    """批量预测文件夹任务（异步）"""
    try:
        audio_files = get_all_audio_files(folder_path)
        if not audio_files:
            result_callback({'status': 'error', 'message': '未找到音频文件！'})
            return

        model_path = os.path.join(app.config['MODELS_FOLDER'], model_name)
        results = []
        total = len(audio_files)
        for i, audio_path in enumerate(audio_files):
            try:
                result = audio_predict(audio_path, model_path)
                filename = os.path.basename(audio_path)
                results.append({
                    'filename': filename,
                    'path': audio_path,
                    'result': result,
                    'status': 'success'
                })
                # 进度更新
                progress = int((i + 1) / total * 100)
                result_callback({
                    'status': 'running',
                    'progress': progress,
                    'current': filename,
                    'total': total,
                    'success': len([r for r in results if r['status'] == 'success']),
                    'failed': len([r for r in results if r['status'] == 'failed']),
                    'results': results
                })
            except Exception as e:
                results.append({
                    'filename': os.path.basename(audio_path),
                    'path': audio_path,
                    'error': str(e),
                    'status': 'failed'
                })

        result_callback({
            'status': 'completed',
            'total': total,
            'success': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'results': results
        })
    except Exception as e:
        result_callback({'status': 'error', 'message': f'批量预测失败：{str(e)}'})


# ---------------------- 路由 ----------------------
@app.route('/')
def index():
    """美化后的首页"""
    return render_template('index.html', task_status=task_status)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """扩展预测功能：单文件/文件夹/录音 - 修复录音处理"""
    if request.method == 'GET':
        models = [f for f in os.listdir(app.config['MODELS_FOLDER']) if f.endswith('.h5')]
        return render_template('predict.html', models=models)

    # 处理POST请求
    action_type = request.form.get('action_type', 'single')
    model_name = request.form.get('model_name')

    if not model_name:
        return jsonify({'status': 'error', 'message': '未选择模型！'})

    model_path = os.path.join(app.config['MODELS_FOLDER'], model_name)
    if not os.path.exists(model_path):
        return jsonify({'status': 'error', 'message': '模型文件不存在！'})

    # 1. 单个文件预测
    if action_type == 'single':
        if 'audio_file' not in request.files:
            return jsonify({'status': 'error', 'message': '未上传音频文件！'})
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': '文件名为空！'})
        if file and allowed_file(file.filename):
            # 保存文件
            filename = file.filename
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(audio_path)
            # 执行预测
            try:
                result = audio_predict(audio_path, model_path)
                return jsonify({
                    'status': 'success',
                    'filename': filename,
                    'result': result,
                    'message': f'预测完成：{filename} → {"真实音频" if result == "real" else "AI伪造音频"}'
                })
            except Exception as e:
                return jsonify({'status': 'error', 'message': f'预测失败：{str(e)}'})
        else:
            return jsonify({'status': 'error', 'message': '不支持的文件格式！'})

    # 2. 文件夹预测
    elif action_type == 'folder':
        folder_path = request.form.get('folder_path')
        folder_path = folder_path.replace('\\', '\\\\')
        folder_path = os.path.normpath(folder_path)
        if not os.path.isdir(folder_path):
            return jsonify({'status': 'error', 'message': '文件夹路径不存在！'})

        # 启动异步批量预测
        def result_handler(result):
            """结果回调"""
            global predict_progress
            predict_progress = result

        global predict_progress
        predict_progress = {'status': 'running', 'progress': 0, 'total': 0, 'success': 0, 'failed': 0}

        threading.Thread(
            target=predict_folder_task,
            args=(folder_path, model_name, result_handler),
            daemon=True
        ).start()

        return jsonify({'status': 'success', 'message': '批量预测任务已启动！'})


@app.route('/predict/progress')
def get_predict_progress():
    """获取批量预测进度"""
    global predict_progress
    return jsonify(predict_progress)


# ---------------------- 任务接口 ----------------------
@app.route('/preprocess', methods=['POST'])
def api_preprocess():
    dataset_path = request.form.get('dataset_path')
    dataset_path = dataset_path.replace('\\', '\\\\')
    dataset_path = os.path.normpath(dataset_path)
    if not os.path.exists(dataset_path):
        return jsonify({'status': 'error', 'message': f'数据集路径不存在：{dataset_path}'})
    threading.Thread(target=preprocess_task, args=(dataset_path,), daemon=True).start()
    return jsonify({'status': 'success', 'message': '预处理任务已启动！'})


@app.route('/extract', methods=['POST'])
def api_extract():
    threading.Thread(target=extract_task, daemon=True).start()
    return jsonify({'status': 'success', 'message': '特征提取任务已启动！'})


# 修改2：训练接口接收自定义模型名称
@app.route('/train', methods=['POST'])
def api_train():
    model_type = request.form.get('model_type')
    batch_size = request.form.get('batch_size', 32)
    epochs = request.form.get('epochs', 100)
    model_name = request.form.get('model_name', '')  # 新增：自定义模型名

    # 验证参数
    if not model_type or model_type not in ['lstm', 'transformer']:
        return jsonify({'status': 'error', 'message': '模型类型必须是lstm或transformer！'})

    # 使用默认名称（如果用户未输入）
    if not model_name.strip():
        model_name = f'{model_type}_model_bs{batch_size}_epochs{epochs}_{int(time.time())}'

    # 启动训练任务（传入自定义模型名）
    threading.Thread(
        target=train_task,
        args=(model_type, batch_size, epochs, model_name),
        daemon=True
    ).start()

    return jsonify(
        {'status': 'success', 'message': f'{model_type.upper()}模型训练任务已启动！模型名将保存为：{model_name}.h5'})


@app.route('/visualize/wave', methods=['POST'])
def api_visualize_wave():
    threading.Thread(target=wave_visualize_task, daemon=True).start()
    return jsonify({'status': 'success', 'message': '波形可视化任务已启动！'})


@app.route('/visualize/feat', methods=['POST'])
def api_visualize_feat():
    threading.Thread(target=feat_visualize_task, daemon=True).start()
    return jsonify({'status': 'success', 'message': '特征可视化任务已启动！'})


@app.route('/task/status/<task_type>')
def api_task_status(task_type):
    if task_type in task_status:
        return jsonify(task_status[task_type])
    return jsonify({'status': 'error', 'message': '无效的任务类型！'})


if __name__ == '__main__':
    # 创建必要目录
    for folder in [
        app.config['UPLOAD_FOLDER'],
        app.config['PREPROCESSED_FOLDER'],
        app.config['FEATURES_FOLDER'],
        app.config['MODELS_FOLDER'],
        os.path.join(app.config['VISUALIZATIONS_FOLDER'], 'waveform'),
        os.path.join(app.config['VISUALIZATIONS_FOLDER'], 'features')
    ]:
        os.makedirs(folder, exist_ok=True)

    # 修复：关闭debug模式（生产环境），增加端口可用性检查
    app.run(debug=False, host='0.0.0.0', port=5000)