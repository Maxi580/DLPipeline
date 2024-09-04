from flask import Flask, render_template, request
import os

ENV_PATH = os.getenv('ENV_PATH')
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def config_page():
    if request.method == 'POST':
        config = {
            'IMAGE_WIDTH': request.form['IMAGE_WIDTH'],
            'IMAGE_HEIGHT': request.form['IMAGE_HEIGHT'],
            'YOLO_WITH_AUGMENTATION': request.form['YOLO_WITH_AUGMENTATION'],
            'YOLO_WITHOUT_AUGMENTATION': request.form['YOLO_WITHOUT_AUGMENTATION'],
            'FASTER_RCNN_WITH_AUGMENTATION': request.form['FASTER_RCNN_WITH_AUGMENTATION'],
            'FASTER_RCNN_WITHOUT_AUGMENTATION': request.form['FASTER_RCNN_WITHOUT_AUGMENTATION'],
            'UNET_WITH_AUGMENTATION': request.form['UNET_WITH_AUGMENTATION'],
            'UNET_WITHOUT_AUGMENTATION': request.form['UNET_WITHOUT_AUGMENTATION'],
            'UNET_BATCH_SIZE': request.form['UNET_BATCH_SIZE'],
            'UNET_NUM_CLASSES': request.form['UNET_NUM_CLASSES'],
            'UNET_NUM_WORKERS': request.form['UNET_NUM_WORKERS'],
            'UNET_EPOCHS': request.form['UNET_EPOCHS'],
            'UNET_LR': request.form['UNET_LR'],
            'FRCNN_MODELS': request.form.getlist('FRCNN_MODELS[]'),
            'YOLO_MODELS': request.form.getlist('YOLO_MODELS[]'),
            'NUMBER_OF_AUGMENTED_IMAGES': request.form['NUMBER_OF_AUGMENTED_IMAGES'],
            'YOLO_EPOCHS': request.form['YOLO_EPOCHS'],
            'FRCNN_EPOCHS': request.form['FRCNN_EPOCHS'],
            'YOLO_BATCH_SIZE': request.form['YOLO_BATCH_SIZE'],
            'FRCNN_BATCH_SIZE': request.form['FRCNN_BATCH_SIZE'],
            'IOU_THRESHOLD': request.form['IOU_THRESHOLD'],
            'AUGMENTATION_PROBABILITY': request.form['AUGMENTATION_PROBABILITY'],
            'MIXING_PROBABILITY': request.form['MIXING_PROBABILITY'],
            'MIXING_FACTOR_LOWER_RANGE': request.form['MIXING_FACTOR_LOWER_RANGE'],
            'MIXING_FACTOR_UPPER_RANGE': request.form['MIXING_FACTOR_UPPER_RANGE'],
            'ENABLE_HORIZONTAL_FLIP': request.form['ENABLE_HORIZONTAL_FLIP'],
            'ENABLE_VERTICAL_FLIP': request.form['ENABLE_VERTICAL_FLIP'],
            'ENABLE_ROTATE': request.form['ENABLE_ROTATE'],
            'ROTATE_LIMIT': request.form['ROTATE_LIMIT'],
            'ENABLE_HUE_SATURATION': request.form['ENABLE_HUE_SATURATION'],
            'HUE_SHIFT_LIMIT': request.form['HUE_SHIFT_LIMIT'],
            'SAT_SHIFT_LIMIT': request.form['SAT_SHIFT_LIMIT'],
            'VAL_SHIFT_LIMIT': request.form['VAL_SHIFT_LIMIT'],
            'ENABLE_BRIGHTNESS_CONTRAST': request.form['ENABLE_BRIGHTNESS_CONTRAST'],
            'BRIGHTNESS_LIMIT': request.form['BRIGHTNESS_LIMIT'],
            'CONTRAST_LIMIT': request.form['CONTRAST_LIMIT'],
            'ENABLE_SHEAR': request.form['ENABLE_SHEAR'],
            'SHEAR_DEGREE_LIMIT': request.form['SHEAR_DEGREE_LIMIT'],
            'ENABLE_GAUSSIAN_BLUR': request.form['ENABLE_GAUSSIAN_BLUR'],
            'GAUSSIAN_BLUR_MINIMUM': request.form['GAUSSIAN_BLUR_MINIMUM'],
            'GAUSSIAN_BLUR_MAX': request.form['GAUSSIAN_BLUR_MAX'],
            'ENABLE_GAUSSIAN_NOISE': request.form['ENABLE_GAUSSIAN_NOISE'],
            'NOISE_VAR_LIMIT': request.form['NOISE_VAR_LIMIT'],
            'RANDOM_GAMMA_LIMIT': request.form['RANDOM_GAMMA_LIMIT'],
            'ENABLE_RANDOM_RAIN': request.form['ENABLE_RANDOM_RAIN'],
            'ENABLE_RANDOM_FOG': request.form['ENABLE_RANDOM_FOG'],
            'ENABLE_RANDOM_SNOW': request.form['ENABLE_RANDOM_SNOW'],
            'ENABLE_RANDOM_SHADOW': request.form['ENABLE_RANDOM_SHADOW'],
            'OPTIMIZER_LEARNING_RATE': request.form['OPTIMIZER_LEARNING_RATE'],
            'OPTIMIZER_MOMENTUM': request.form['OPTIMIZER_MOMENTUM'],
            'OPTIMIZER_WEIGHT_DECAY': request.form['OPTIMIZER_WEIGHT_DECAY'],
            'SCHEDULER_STEP_SIZE': request.form['SCHEDULER_STEP_SIZE'],
            'SCHEDULER_GAMMA': request.form['SCHEDULER_GAMMA'],
            'EARLY_STOPPING_PATIENCE': request.form['EARLY_STOPPING_PATIENCE'],
            'EARLY_STOPPING_MIN_DELTA': request.form['EARLY_STOPPING_MIN_DELTA'],
        }
        #  Add commas between arrays and format them as string
        if isinstance(config['YOLO_MODELS'], list):
            yolo_models = ','.join(config['YOLO_MODELS'])
        else:
            yolo_models = config['YOLO_MODELS']
        config['YOLO_MODELS'] = yolo_models
        if isinstance(config['FRCNN_MODELS'], list):
            frcnn_models = ','.join(config['FRCNN_MODELS'])
        else:
            frcnn_models = config['FRCNN_MODELS']
        config['FRCNN_MODELS'] = frcnn_models

        env_path = os.path.join(ENV_PATH)
        existing_config = {}
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        existing_config[key] = value

        existing_config.update(config)

        with open(env_path, 'w') as f:
            for key, value in existing_config.items():
                f.write(f"{key}={value}\n")

        return 'Configuration updated in .env file!'
    return render_template('config_form.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
