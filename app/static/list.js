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

document.addEventListener('DOMContentLoaded', function() {
    const listButton = document.querySelector('button');
    if (listButton) {
        listButton.addEventListener('click', listContents);
    } else {
        console.error('List contents button not found');
    }
});

window.listContents = listContents;
window.downloadFile = downloadFile;

console.log('list.js loaded');
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded');
    console.log('volumeId element:', document.getElementById('volumeId'));
    console.log('path element:', document.getElementById('path'));
});