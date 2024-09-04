import subprocess
import os

# Script to copy local images for training onto data_volume container

def docker_cp_to_container(local_path, container_id, container_path):
    local_path = os.path.normpath(local_path)

    command_a_t = f'docker cp {local_path}/annotations/train {container_id}:{container_path}/labels'
    command_a_v = f'docker cp {local_path}/annotations/val {container_id}:{container_path}/labels'
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


container_id = "2d85d23aead1c312901242c1633e3e0e69e6bb03f96f7edb241d3147a7f8f9ca"  # Id of data_volume Container
local_path = r"C:\Users\maxie\Desktop\FruitData"  # Path to local images for training, need to be in correct directories (see above)
container_path = "/data"

if __name__ == '__main__':
    docker_cp_to_container(local_path, container_id, container_path)
