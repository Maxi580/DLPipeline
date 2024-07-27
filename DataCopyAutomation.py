import subprocess
import os


def docker_cp_to_container(local_path, container_id, container_path):
    local_path = os.path.normpath(local_path)

    command_a_t = f'docker cp {local_path}/annotations/train {container_id}:{container_path}/annotations'
    command_a_v = f'docker cp {local_path}/annotations/val {container_id}:{container_path}/annotations'
    command_i_t = f'docker cp {local_path}/images/train {container_id}:{container_path}/images'
    command_i_v = f'docker cp {local_path}/images/val {container_id}:{container_path}/images'

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


container_id = "d64b914479ca4be5916d9dcd870ab5e6127bd398b4e5fc19351ffafa6cc40b3a"
local_path = r"C:/Users/maxie/Desktop/LicensePlateData"
container_path = "/data"

if __name__ == '__main__':
    docker_cp_to_container(local_path, container_id, container_path)
