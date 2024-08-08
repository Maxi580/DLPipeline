# Annotations
* Preprocessing currently expects xml (pascalvoc), json (coco) or txt (yolo)


# Steps
* Configure training over web Interface
* Load images into data_volume (over script)
* preprocess pictures
* augment if you want
* train model
* load inference pictures and model onto inference_volume, start container 


# Set up Docker Container
* `docker-compose build --no-cache`
* `docker-compose down -v`
* `docker-compose up --build`
* Reset Single Services 
* `docker-compose build service-name`
* `docker-compose up -d --force-recreate service-name`

# Copy Data onto Data Volume
* `docker cp /path/to/local/image containerId:/path/in/container/` (
 The Path is /data/[images/annotations] (There are two helper scripts, two support image copying for training and inference)