# Reset Docker Environment
* docker-compose build --no-cache
* docker-compose down -v
* docker-compose up --build

# Copy Data onto Data Volume
* docker cp /path/to/local/image containerId:/path/in/container/ (
* The Path is /data/[images/annotations] (Definitely Leave out Train or test path as only then it works,
                                          I have no Idea why though)