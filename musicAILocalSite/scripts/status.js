function populate_training_songs_table() {
    $.get('http://68.227.63.30:3000/is_model_running', {},
            function (data) {
                b = document.getElementById('start_training');
                b.hidden = false;
                if (data === 'True') {
                    b.value = 'Training is already Running!';
                    b.disabled = true;
                }
            }, 'text');

    $.get('http://68.227.63.30:3000/input_files', {},
            function (data) {
                console.log('Got song population table:', data);

                var t = document.getElementById('training_songs_table');
                data['input_files'].forEach(function (fn) {
                    var r = t.insertRow(-1);
                    var c = r.insertCell();
                    c.innerHTML = fn;
                    c.className = 'input_table_item';
                });
            }, 'json');
}

function train_model() {
    console.log('Starting model script');
    $.get('http://68.227.63.30:3000/train_model_start', {},
            function (data) {
                if (data === 'started') {
                    document.getElementById('model_output_table').hidden = false;
                    b = document.getElementById('start_training');
                    b.value = 'Training Started!';
                    b.disabled = true;
                    get_training_status();
                } else {
                    alert('Failed: ' + data);
                }
            }, 'text');
}

function get_training_status() {
    var t = document.getElementById('model_output_table');

    $.get('http://68.227.63.30:3000/train_model_status', {},
            function (data) {

                data['script_output_new'].forEach(function (l) {
                    var r = t.insertRow(-1);
                    var c = r.insertCell();
                    c.innerHTML = l;
                });

                if (data['script_running']) {
                    setTimeout(get_training_status, 200);
                } else {
                    document.getElementById('generated_header').hidden = false;
                    document.getElementById('upload_result_file_input').value = data['output_file'];
                    document.getElementById('upload_result_form').hidden = false;
                }
            }, 'json');
}