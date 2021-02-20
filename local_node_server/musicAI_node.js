// Ex: curl http://localhost:3000/train_model_status


const http = require('http');
const fs = require('fs');
const express = require('express')

const port = 3000;

const app = express()

app.get('/input_files', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.statusCode = 200;

  var j = {
    input_files: [],
  }

  fs.readdirSync('/home/dp/NetBeansProjects/musicAILocalSite/data_files').forEach(fn => {
    j['input_files'].push(fn);
  });

  res.json(j)
})

var dequeue = require('./dequeue');
var model_out_buffer = new dequeue.Dequeue()
var model_running = false
const model_input_glob = "/home/dp/NetBeansProjects/musicAILocalSite/data_files/*.mid"
const model_output_file = "/home/dp/NetBeansProjects/musicAILocalSite/model_output.mp3"

app.get('/train_model_start', function (req, res) {
  res.set({
    'Access-Control-Allow-Origin': '*',
  });
  res.statusCode = 200;


  if (!model_running) {
    model_out_buffer.clear()
    model_running = true;

    const spawn = require("child_process").spawn;
    const pythonProcess = spawn('python3', ["/home/dp/Desktop/cs425_music_ai/ml_model/main.py",
      "--input_files_glob", model_input_glob,
      "--output_file", model_output_file,
      "--num_epochs", 5,
      "--output_notes", 50]);
    res.write('started');

    console.log(`Training script started!`);

    pythonProcess.stdout.on('data', (data) => {
      var data_arr = data.toString().split(/\r?\n/);
      for (var i = 0; i < data_arr.length; i++) {
        if (!data_arr[i] || !data_arr[i].trim()) continue;
        console.log(data_arr[i]);
        model_out_buffer.addBack(data_arr[i]);
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`Training script ended with code ${code}`);
      model_running = false;
    });
  } else {
    res.write('Cannot start multiple training scripts!');
  }
  res.end();
})

app.get('/train_model_status', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  res.statusCode = 200;

  var j = {
    script_output_new: [],
    script_running: model_running
  }

  for (var i = 0; i < Math.min(model_out_buffer.size(), 20); i++)
    j['script_output_new'].push(model_out_buffer.removeFront());

  if (!j['script_running'])
    j['output_file'] = model_output_file

  res.json(j)
})


app.get('/is_model_running', function (req, res) {
  res.set({
    'Access-Control-Allow-Origin': '*',
  });
  res.statusCode = 200;

  if (model_running) res.write('True');
  else res.write('False');

  res.end();
})

app.listen(port)

