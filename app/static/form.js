document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('configForm');

    form.addEventListener('submit', function(e) {
        e.preventDefault();

        const formData = new FormData(form);

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            console.log('Server response:', data);
            alert('Configuration submitted!\n' + data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while submitting the configuration.');
        });
    });
});
