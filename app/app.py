from flask import Flask, render_template, request
import os

ENV_PATH = os.getenv('ENV_PATH')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def config_page():
    if request.method == 'POST':
        config = {
            'CREATE_YOLO_MODEL': request.form['CREATE_YOLO_MODEL'],
            'PIXMIX_AUGMENTATION': request.form['PIXMIX_AUGMENTATION'],
            'CLASSES': request.form['CLASSES'],
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
    app.run(host='0.0.0.0', port=5000, debug=False)
