import subprocess
import os


def docker_cp_to_container(local_path, container_id, container_path):
    local_path = os.path.normpath(local_path)

    command_a_t = f'docker cp "{local_path}"/annotations/train {container_id}:{container_path}/annotations'
    command_a_v = f'docker cp "{local_path}"/annotations/val {container_id}:{container_path}/annotations'
    command_i_t = f'docker cp "{local_path}"/images/train {container_id}:{container_path}/images'
    command_i_v = f'docker cp "{local_path}"/images/val {container_id}:{container_path}/images'

    try:
        subprocess.run(command_a_t, check=True, shell=True, capture_output=True, text=True)
        subprocess.run(command_a_v, check=True, shell=True, capture_output=True, text=True)
        subprocess.run(command_i_t, check=True, shell=True, capture_output=True, text=True)
        subprocess.run(command_i_v, check=True, shell=True, capture_output=True, text=True)
        print(f"Successfully copied {local_path} to {container_id}:{container_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error copying file: {e}")
        print(f"Error output: {e.stderr}")
        return False


container_id = "0dee0d72218ffec7300e85f4d1e3790120a5cdba044d14e4e7a82d1be2ac7fc7"
local_path = r"C:\Users\a880902\OneDrive - Eviden\Desktop\Datasetss\LicensePlates"
container_path = "/data"

if __name__ == '__main__':
    docker_cp_to_container(local_path, container_id, container_path)