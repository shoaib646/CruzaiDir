function generateRoutes() {
        // Show loading message
        document.getElementById('loadingMessage').style.display = 'block';

        // Submit the form
        document.getElementById('optimalForm').submit();
    }

    var socket = new WebSocket('ws://' + window.location.host + '/ws/OptimalPath/');

    socket.onmessage = function (e) {
        var data = JSON.parse(e.data);

        // Hide loading message
        document.getElementById('loadingMessage').style.display = 'Generating Routes';

        // Update the textarea with the received routes
        document.getElementById('routesTextarea').value = data.routes;
    };

    socket.onclose = function (event) {
        console.error('WebSocket closed:', event);
    };