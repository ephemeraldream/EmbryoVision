

function startNeuralNetwork(){

    fetch("{% url 'deepapp:start_neural_network' %}", {
        method:"POST",
        headers: {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
        .then((response => response.json()))
        .then((data => {
            console.log(data)
        }))
        .catch(error => console.error('We have an ERROR', error));
}
function putLabels() {
    fetch("{% url 'deepapp:put_labels' %}", {
        method: 'POST',
        headers: {
            'X-CSRFToken': getCookie('csrftoken'),
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {

        console.log(data);
    })
    .catch(error => console.error('Error:', error));
}
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}