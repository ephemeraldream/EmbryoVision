import * as d3 from 'd3'



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
        if (data){
            document.getElementById('classification-prediction').innerText = data.classification_pred.join(', ');
            document.getElementById('regression-prediction').innerText = data.regression_pred.join(', ');
            document.getElementById('hole-prediction').innerText = data.hole_pred.toString();
        }
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
function test() {
  document.getElementById("trash").innerHTML = "Hello World";
}