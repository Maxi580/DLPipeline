document.addEventListener('DOMContentLoaded', (event) => {
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

        fileItems.forEach((item, index) => {
            const file = item.file;  // Assuming you stored the File object on the element
            formData.append(`files[]`, file, file.name);
        });

        fetch('http://127.0.0.1:5000/upload-to-docker', {
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
});