const socket = io();

const ctx = document.getElementById('myChart').getContext('2d');
const myChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: '实时数据',
            data: [],
            borderColor: 'rgba(75, 192, 192, 1)',
            borderWidth: 1,
            fill: false
        }]
    },
    options: {
        scales: {
            x: {
                type: 'linear',
                position: 'bottom'
            }
        }
    }
});

socket.on('update_chart', function(data) {
    myChart.data.labels.push(data.time);
    myChart.data.datasets[0].data.push(data.value);
    myChart.update();
});

function sendInput() {
    const userInput = document.getElementById('user-input').value;
    socket.emit('user_input', userInput);
}