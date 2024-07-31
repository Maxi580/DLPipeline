from flask import Flask, render_template, request
import os

ENV_PATH = os.getenv('ENV_PATH')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def config_page():
    if request.method == 'POST':
        config = {
            'PIXMIX_AUGMENTATION': request.form['PIXMIX_AUGMENTATION'],
            'PIXMIX_AUGMENTATION_PROBABILITY': request.form['PIXMIX_AUGMENTATION_PROBABILITY'],
            'PIXMIX_MIXING_PROBABILITY': request.form['PIXMIX_MIXING_PROBABILITY'],
            'MIXING_FACTOR_LOWER_RANGE': request.form['MIXING_FACTOR_LOWER_RANGE'],
            'MIXING_FACTOR_UPPER_RANGE': request.form['MIXING_FACTOR_UPPER_RANGE'],
            'MODEL_WITHOUT_AUGMENTATION': request.form['MODEL_WITHOUT_AUGMENTATION'],
            'CREATE_YOLO_MODEL': request.form['CREATE_YOLO_MODEL'],
            'YOLO_MODEL': request.form['YOLO_MODEL'],
            'NUMBER_OF_AUGMENTED_IMAGES': request.form['NUMBER_OF_AUGMENTED_IMAGES'],
            'EPOCHS': request.form['EPOCHS'],
            'BATCH_SIZE': request.form['BATCH_SIZE'],
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
            'RANDOM_GAMMA_LIMIT': request.form['RANDOM_GAMMA_LIMIT'],
            'ENABLE_RANDOM_RAIN': request.form['ENABLE_RANDOM_RAIN'],
            'ENABLE_RANDOM_FOG': request.form['ENABLE_RANDOM_FOG'],
            'ENABLE_RANDOM_SNOW': request.form['ENABLE_RANDOM_SNOW'],
            'ENABLE_RANDOM_SHADOW': request.form['ENABLE_RANDOM_SHADOW'],
            'ENABLE_RANDOM_SUNFLARE': request.form['ENABLE_RANDOM_SUNFLARE'],
        }
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
