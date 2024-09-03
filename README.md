# Annotations
* Preprocessing currently expects xml (pascalvoc), json (coco) or txt (yolo)

# Set up Docker Container
* `docker-compose build --no-cache` (Try Again on Connection Timeout)
* `docker-compose down -v`
* `docker-compose up --build`
* Reset Single Services 
* `docker-compose build service-name`
* `docker-compose up -d --force-recreate service-name`

# Copy Data onto Data Volume
* `docker cp /path/to/local/image containerId:/path/in/container/` (
 The Path is (and has to be) /data/[images/annotations]/[train/val] 
* (There are two helper scripts, two support image copying for training and inference)