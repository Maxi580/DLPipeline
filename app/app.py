import docker
from flask import Flask, render_template, request, jsonify
import subprocess
from flask_cors import CORS
import os

ENV_PATH = os.getenv('ENV_PATH')

app = Flask(__name__)

CORS(app)
app.config['UPLOAD_FOLDER'] = '/data/tmp'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def config_page():
    if request.method == 'POST':
        config = {
            'MODEL_WITHOUT_AUGMENTATION': request.form['MODEL_WITHOUT_AUGMENTATION'],
            'CREATE_YOLO_MODEL': request.form['CREATE_YOLO_MODEL'],
            'YOLO_MODEL': request.form['YOLO_MODEL'],
            'NUMBER_OF_AUGMENTED_IMAGES': request.form['NUMBER_OF_AUGMENTED_IMAGES'],
            'EPOCHS': request.form['EPOCHS'],
            'BATCH_SIZE': request.form['BATCH_SIZE'],
            'PIXMIX_AUGMENTATION': request.form['PIXMIX_AUGMENTATION'],
            'PIXMIX_AUGMENTATION_PROBABILITY': request.form['PIXMIX_AUGMENTATION_PROBABILITY'],
            'PIXMIX_MIXING_PROBABILITY': request.form['PIXMIX_MIXING_PROBABILITY'],
            'PIXMIX_MIXING_FACTOR_LOWER_RANGE': request.form['MIXING_FACTOR_LOWER_RANGE'],
            'PIXMIX_MIXING_FACTOR_UPPER_RANGE': request.form['MIXING_FACTOR_UPPER_RANGE'],
            'PIXMIX_ENABLE_HORIZONTAL_FLIP': request.form['PIXMIX_ENABLE_HORIZONTAL_FLIP'],
            'PIXMIX_ENABLE_VERTICAL_FLIP': request.form['PIXMIX_ENABLE_VERTICAL_FLIP'],
            'PIXMIX_ENABLE_ROTATE': request.form['PIXMIX_ENABLE_ROTATE'],
            'PIXMIX_ROTATE_LIMIT': request.form['PIXMIX_ROTATE_LIMIT'],
            'PIXMIX_ENABLE_HUE_SATURATION': request.form['PIXMIX_ENABLE_HUE_SATURATION'],
            'PIXMIX_HUE_SHIFT_LIMIT': request.form['PIXMIX_HUE_SHIFT_LIMIT'],
            'PIXMIX_SAT_SHIFT_LIMIT': request.form['PIXMIX_SAT_SHIFT_LIMIT'],
            'PIXMIX_VAL_SHIFT_LIMIT': request.form['PIXMIX_VAL_SHIFT_LIMIT'],
            'PIXMIX_ENABLE_BRIGHTNESS_CONTRAST': request.form['PIXMIX_ENABLE_BRIGHTNESS_CONTRAST'],
            'PIXMIX_BRIGHTNESS_LIMIT': request.form['PIXMIX_BRIGHTNESS_LIMIT'],
            'PIXMIX_CONTRAST_LIMIT': request.form['PIXMIX_CONTRAST_LIMIT'],
            'PIXMIX_ENABLE_SHEAR': request.form['PIXMIX_ENABLE_SHEAR'],
            'PIXMIX_SHEAR_DEGREE_LIMIT': request.form['PIXMIX_SHEAR_DEGREE_LIMIT'],
            'PIXMIX_ENABLE_GAUSSIAN_BLUR': request.form['PIXMIX_ENABLE_GAUSSIAN_BLUR'],
            'PIXMIX_GAUSSIAN_BLUR_MINIMUM': request.form['PIXMIX_GAUSSIAN_BLUR_MINIMUM'],
            'PIXMIX_GAUSSIAN_BLUR_MAX': request.form['PIXMIX_GAUSSIAN_BLUR_MAX'],
            'PIXMIX_ENABLE_GAUSSIAN_NOISE': request.form['PIXMIX_ENABLE_GAUSSIAN_NOISE'],
            'PIXMIX_NOISE_VAR_LIMIT': request.form['PIXMIX_NOISE_VAR_LIMIT'],
            'PIXMIX_RANDOM_GAMMA_LIMIT': request.form['PIXMIX_RANDOM_GAMMA_LIMIT'],
            'PIXMIX_ENABLE_RANDOM_RAIN': request.form['PIXMIX_ENABLE_RANDOM_RAIN'],
            'PIXMIX_ENABLE_RANDOM_FOG': request.form['PIXMIX_ENABLE_RANDOM_FOG'],
            'PIXMIX_ENABLE_RANDOM_SNOW': request.form['PIXMIX_ENABLE_RANDOM_SNOW'],
            'PIXMIX_ENABLE_RANDOM_SHADOW': request.form['PIXMIX_ENABLE_RANDOM_SHADOW'],
            'PIXMIX_ENABLE_RANDOM_SUNFLARE': request.form['PIXMIX_ENABLE_RANDOM_SUNFLARE'],
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


@app.route('/files', methods=['GET'])
def file_management_page():
    return render_template('file_management.html')


@app.route('/upload-to-docker', methods=['POST'])
def upload_to_docker():
    if 'files[]' not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    files = request.files.getlist('files[]')
    container_id = request.form.get('containerId')
    container_path = request.form.get('containerPath')

    if not container_id or not container_path:
        return jsonify({"success": False, "error": "Missing container ID or path"})

    try:
        for file in files:
            if file and file.filename:
                temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                try:
                    file.save(temp_file_path)
                    cmd = f"docker cp {temp_file_path} {container_id}:{container_path}"
                    subprocess.run(cmd, shell=True, check=True)
                finally:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

        return jsonify({"success": True, "message": "Files uploaded successfully"})
    except subprocess.CalledProcessError as e:
        return jsonify({"success": False, "error": str(e)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/list-docker-contents', methods=['POST'])
def list_docker_contents():
    volumeId = request.form.get('volumeId')
    path = request.form.get('path', '/')

    if not volumeId:
        return jsonify({"success": False, "error": "Volume name is required"}), 400

    try:
        client = docker.from_env()

        # Ensure the alpine image is pulled
        try:
            client.images.get('alpine:latest')
        except docker.errors.ImageNotFound:
            client.images.pull('alpine:latest')

        # Create a temporary container to access the volume
        container = client.containers.create(
            'alpine',
            'ls -la ' + path,
            volumes={volumeId: {'bind': '/mnt/volume', 'mode': 'ro'}},
            working_dir='/mnt/volume'
        )

        # Start the container
        container.start()

        # Wait for the container to finish and get the output
        result = container.wait()
        output = container.logs().decode('utf-8')

        # Remove the temporary container
        container.remove()

        if result['StatusCode'] != 0:
            return jsonify({"success": False, "error": f"Error listing contents: {output}"}), 500

        # Parse the output
        lines = output.split('\n')
        contents = []
        for line in lines[1:]:  # Skip the first line (total)
            if line.strip():
                parts = line.split()
                if len(parts) >= 9:
                    file_type = 'd' if parts[0].startswith('d') else 'f'
                    name = ' '.join(parts[8:])
                    contents.append({"name": name, "type": file_type})

        return jsonify({"success": True, "contents": contents})

    except Exception as e:
        app.logger.error(f"Error listing Docker volume contents: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
