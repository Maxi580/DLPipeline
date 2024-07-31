import subprocess
import os


def docker_cp_images_to_container(local_path, container_id, container_path):
    local_path = os.path.normpath(local_path)

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    copied_count = 0
    error_count = 0
    for root, dirs, files in os.walk(local_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                local_file_path = os.path.join(local_path, file)
                container_file_path = os.path.join(container_path, file)

                copy_command = f'docker cp "{local_file_path}" {container_id}:{container_file_path}'
                print(copy_command)
                try:
                    subprocess.run(copy_command, check=True, shell=True, capture_output=True, text=True)
                    print(f"Successfully copied {local_file_path} to {container_id}:{container_file_path}")
                    copied_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"Error copying file {local_file_path}: {e}")
                    print(f"Error output: {e.stderr}")
                    error_count += 1

    print(f"\nCopying complete. Successfully copied {copied_count} images.")
    if error_count > 0:
        print(f"Encountered errors with {error_count} files.")

    return copied_count > 0


container_id = "69aebefcfe4e1e59c02790e097a954bd795dec77980382517196e1a0c7716419"
local_path = r"C:\Users\maxie\Desktop\inference\input"
container_path = "/inference/input_images"

if __name__ == '__main__':
    docker_cp_images_to_container(local_path, container_id, container_path)
