document.addEventListener('DOMContentLoaded', (event) => {
    // Upload to Docker
    const uploadToDockerBtn = document.getElementById('uploadToDocker');
    if (uploadToDockerBtn) {
        uploadToDockerBtn.addEventListener('click', uploadToDocker);
    } else {
        console.error('Upload to Docker button not found');
    }

    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const fileList = document.getElementById('fileList');

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    fileInput.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });

    function handleFiles(files) {
        for (let file of files) {
            addFileToList(file);
        }
    }

    function addFileToList(file) {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.file = file;
        fileItem.innerHTML = `
            <span>${file.name}</span>
            <button onclick="removeFile(this)">Remove</button>
        `;
        fileList.appendChild(fileItem);
    }

    window.removeFile = function(button) {
        button.closest('.file-item').remove();
    }

    function uploadToDocker() {
        const containerIdElement = document.getElementById('containerId');
        const containerPathElement = document.getElementById('containerPath');

        if (!containerIdElement || !containerPathElement) {
            console.error('Container ID or Container Path input not found');
            alert('Error: Container ID or Container Path input not found');
            return;
        }

        const containerId = containerIdElement.value;
        const containerPath = containerPathElement.value;
        const fileItems = document.querySelectorAll('.file-item');

        if (!containerId || !containerPath || fileItems.length === 0) {
            alert('Please provide container ID, container path, and upload at least one file.');
            return;
        }

        const formData = new FormData();
        formData.append('containerId', containerId);
        formData.append('containerPath', containerPath);

        fileItems.forEach((item) => {
            const file = item.file;  // Assuming you stored the File object on the element
            formData.append(`files[]`, file, file.name);
        });

        fetch('/upload-to-docker', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Files uploaded to Docker container successfully!');
            } else {
                alert('Error uploading files: ' + data.error);
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('An error occurred while uploading files.');
        });
    }

    // List Docker Volume Contents
    const listContentsBtn = document.querySelector('.containerbrowser button');
    if (listContentsBtn) {
        listContentsBtn.addEventListener('click', listContents);
    } else {
        console.error('List contents button not found');
    }

    function listContents() {
        const volumeIdElement = document.getElementById('volumeId');
        const pathElement = document.getElementById('path');

        if (!volumeIdElement || !pathElement) {
            console.error('Volume ID or Path input not found');
            alert('Error: Volume ID or Path input not found');
            return;
        }

        const volumeId = volumeIdElement.value;
        const path = pathElement.value || '/';

        if (!volumeId) {
            alert('Please enter a Volume ID');
            return;
        }

        const formData = new FormData();
        formData.append('volumeId', volumeId);
        formData.append('path', path);

        fetch('/list-docker-contents', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayContents(data.contents, path);
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while listing contents.');
        });
    }

    function displayContents(contents, path) {
        const contentsList = document.getElementById('contents-list');
        contentsList.innerHTML = '';

        contents.forEach(item => {
            const itemElement = document.createElement('div');
            itemElement.className = 'content-item';
            itemElement.innerHTML = `
                <span>${item.type === 'd' ? '[DIR]' : '[FILE]'} ${item.name}</span>
            `;
            contentsList.appendChild(itemElement);
        });
    }
});
