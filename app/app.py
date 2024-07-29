from flask import Flask, render_template, request
import os

ENV_PATH = os.getenv('ENV_PATH')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def config_page():
    if request.method == 'POST':
        config = {
            'PIXMIX_AUGMENTATION': request.form['PIXMIX_AUGMENTATION'],
            'WITHOUT_AUGMENTATION': request.form['PIXMIX_AUGMENTATION'],
            'CREATE_YOLO_MODEL': request.form['CREATE_YOLO_MODEL'],
            'NUMBER_OF_AUGMENTATION_RUNS': request.form['NUMBER_OF_AUGMENTATION_RUNS'],
            'AUGMENTATIONS': request.form['AUGMENTATIONS'],
            'EPOCHS': request.form['EPOCHS'],
            'BATCH_SIZE': request.form['BATCH_SIZE'],
        }

        env_path = os.path.join(ENV_PATH)

        # Read existing content
        existing_config = {}
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        existing_config[key] = value

        # Update with new values
        existing_config.update(config)

        # Write updated content back to file
        with open(env_path, 'w') as f:
            for key, value in existing_config.items():
                f.write(f"{key}={value}\n")

        return 'Configuration updated in .env file!'
    return render_template('config_form.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
