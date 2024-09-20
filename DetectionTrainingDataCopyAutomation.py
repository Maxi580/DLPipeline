import subprocess
import os

# Script to copy local images for training onto data_volume container

def docker_cp_to_container(local_path, container_id, container_path):
    local_path = os.path.normpath(local_path)

    command_a_t = f'docker cp {local_path}/train/images {container_id}:{container_path}/images/train'
    command_a_v = f'docker cp {local_path}/train/labels {container_id}:{container_path}/labels/train'
    command_i_t = f'docker cp {local_path}/test/images {container_id}:{container_path}/images/val'
    command_i_v = f'docker cp {local_path}/test/labels {container_id}:{container_path}/labels/val'

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


container_id = "44147262d06d2f45447e677780a8f9a169666c4f4634995d0736dc09243c2281"  # Id of data_volume Container
local_path = r"C:\Users\maxie\Desktop\TreeDataset"  # Path to local images for training, need to be in correct directories (see above)
container_path = "/data"

if __name__ == '__main__':
    docker_cp_to_container(local_path, container_id, container_path)
