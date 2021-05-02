/**
 * Main file for backend NodeJS server of musicAI.
 * Provides endpoints for UI and other backend scripts to start training jobs and get their status.
 * Automatically handles updating and maintaining job status's in the SQL database.
 * Also maintains an FTP server with the https://musicai.app website to allow for uploading/downloading of files.
 * 
 * Usage: 1. Start server: `nodejs musicAI_node.js`
 *        2. curl endpoints: `curl 'http://localhost:3000/endpoint?var=val&var2=val' | js`
 */


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Required packages and libraries.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


const https = require('https');
const fs = require('fs');
const Ftp = require('ftp');
const findit = require('findit');
const sql = require('sync-mysql')
const { NONAME } = require('dns');
const { exit } = require('process');
const dequeue = require('./dequeue');
const express = require('express');
const twitter = require('twitter-lite');
const youtube_downloader = require("youtube-mp3-downloader");
var crypto = require('crypto');
const { stringify } = require('qs');
const { find } = require('async');


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Setup and connect to external servers.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


// Whether or not to connect to ftp and sql servers.
const ftp_connect = true;
const sql_connect = true;

// Setup server on port 3000 with self signed SSL certificate.
const app = express()
const port = 3000;
const options = {
  key: fs.readFileSync('server-key.pem'),
  cert: fs.readFileSync('server-cert.pem')
};
var httpsServer = https.createServer(options, app);

// Setup FTP connection with https://musicai.app
const ftpClient = new Ftp();
const ftpUploadDir = 'public_html/results/';
const ftpUploadDirYoutube = 'public_html/youtube/downloads/';
if (ftp_connect) {
  ftpClient.connect({
    'host': 'cpanel.freehosting.com',
    'user': 'upload@musicai.app',
    'password': 'upload24!'
  });

  // Get list of generated *.mp3 files that are on website. Used to verify if results have been uploaded.
  var ftpUploadedLst = []
  ftpClient.on('ready', function () {
    ftpClient.list(ftpUploadDir, false, function (error, listing) {
      listing.forEach(function (l) {
        if (l.name.endsWith('.mp3')) {
          ftpUploadedLst.push(l.name);
        }
      });
      console.log('Got Uploaded Files From FTP!');
    });
    console.log('Connected to FTP!');
  });
} else {
  console.log("FTP connection skipped!");
}

// Setup connection with main MySQL database. This database contains information on jobs and users.
const db_con = sql_connect ? new sql({
  host: 'freedb.tech',
  user: 'freedbtech_db',
  password: 'db',
  database: 'freedbtech_musicaidb'
}) : null;
console.log(sql_connect ? 'Connected to MySQL!' : "MySQL connection skipped!");


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Define variables and functions to run model training script.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


// The |job_queue| is used to keep a queue of jobs that need to be scheduled for execution. 
var job_queue = new dequeue.Dequeue();
var job_running = null;
var job_last_done = null;

// The |model_out_buffer| temporarily holds the output of the training script. 
// As data comes in, it is appended to the back. 
// Periodically, the data is removed from the front and saved to a more permanent |model_out_file| file and live streamed to the front end.
var model_out_buffer = new dequeue.Dequeue()
var model_out_file_lock = false;
var model_out_file = null;

// Default training params.
var default_num_epochs = 100;
var default_output_notes = 50;

// Helper function to write data to the output file.
function append_to_model_file(content) {
  if (!model_out_file) return;
  fs.appendFileSync(model_out_file, content + '\n', err => {
    if (err) {
      console.error('Failed to write: ' + err);
    }
  });
}

// Helper function to begin the next job in the queue if there is no job currently running or the current one is done.
function begin_next_job() {
  if (job_running) return;

  if (job_last_done) {
    // Update the status of the done job in the Database.
    try {
      db_con.query(`UPDATE Jobs SET status = "Done" WHERE id='${job_last_done}'`);
    } catch (error) {
      console.log('FATAL: Could not mark job |' + job_last_done + '| as completed!');
      throw error;
    }
  }

  if (model_out_file) {
    // Move data from the buffer to the output file. Make sure to obtain a lock to the file.
    while (model_out_file_lock);
    model_out_file_lock = true;
    while (!model_out_buffer.isEmpty()) {
      append_to_model_file(model_out_buffer.removeFront());
    }
    model_out_file = null;
    model_out_file_lock = false;
  }

  // No job in the queue.
  if (job_queue.isEmpty()) return;

  // Start job at the front of the queue.
  j = job_queue.removeFront();
  job_running = j['id'];
  model_out_file = j['out_file'];
  model_param_num_epochs = j['num_epochs'] || default_num_epochs;
  model_param_output_notes = j['output_size'] || default_output_notes;
  const model_input_glob = model_out_file.replace('_log.log', '_input_*');
  const model_output_music_file = model_out_file.replace('_log.log', '_output.mp3');

  // Clear buffer and remove any old output file.
  model_out_buffer.clear();
  fs.unlink(model_out_file, (err) => { });

  // Start training script.
  const spawn = require("child_process").spawn;
  const model_train_process = spawn('python3.8', ["/home/dp/Desktop/cs425_music_ai/ml_model/main_midi_only.py",
    "--input_files_glob", model_input_glob,
    "--output_file", model_output_music_file,
    "--num_epochs", model_param_num_epochs,
    "--output_notes", model_param_output_notes]);

  // Process standard out data from training script by adding it the |model_out_buffer|.
  function process_data(data) {
    var data_arr = data.toString().split(/\r?\n/);
    for (var i = 0; i < data_arr.length; i++) {
      if (!data_arr[i] || !data_arr[i].trim()) continue;
      console.log(data_arr[i]);
      model_out_buffer.addBack(data_arr[i]);
    }
  }
  model_train_process.stdout.on('data', (data) => {
    process_data(data)
  });
  model_train_process.stderr.on('data', (data) => {
    process_data(data)
  });

  // On completion of the training job, update the status in the database and mark the job as done.
  model_train_process.on('close', (code) => {
    log_str = '';
    if (code != 0) {
      log_str = `ERROR: Training script ended with code ${code}!`;
      db_con.query(`UPDATE Jobs SET status = "Error" WHERE id='${job_running}'`);
    } else {
      log_str = `SUCCESS: Training script ended!`;
    }
    console.log(log_str);
    model_out_buffer.addBack('');
    model_out_buffer.addBack(log_str);

    job_last_done = job_running;
    job_running = null;

    begin_next_job();
  });

  // Update database table by marking job as "Running".
  try {
    db_con.query(`UPDATE Jobs SET status = "Running" WHERE id='${job_running}'`);
    console.log(`Training script started for job ${job_running}`);
  } catch (error) {
    console.log('FATAL: Could not mark job |' + job_last_done + '| as running!');
    throw error;
  }
}


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Endpoints for dealing with jobs. Allows users to get information on and updates jobs.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


/*
  Endpoint to the information on a specific job.
  @params: |job_id|: Id of job
  @output: json:  |err|: If there was any error
                  |err_msg|: Error message if there was an error.
                  |job|: Json with detail of job.
*/
app.get('/job', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  const job_id = req.query.job_id;

  // Return json.
  var j = {
    err: false,
    err_msg: '',
    job: {},
  };

  try {
    // Get information from the database.
    var q = `SELECT j.id, u.username, j.created, j.status, j.output FROM Jobs j INNER JOIN Users u on u.id = j.user WHERE j.id="${job_id}"`;

    var q_r = db_con.query(q);
    if (q_r.length == 1) {
      q_r = q_r[0];
      var output_file = q_r.output;
      var j_status = q_r.status;

      if (j_status == 'Created') {
        if (output_file) {
          console.log('WARN: Job |' + job_id + '| should not have output file |' + output_file + '|');
          output_file = null;
        }
      } else if (j_status == 'Queued') {
        if (output_file) {
          // If job status is queued, ensure that job is in queue. If it is not, add it.
          found_in_queue = (job_running == job_id);
          for (i = 0; i < job_queue.size(); i++) {
            if (job_queue.peek(i)['id'] === job_id) {
              found_in_queue = true;
              break;
            }
          }
          if (!found_in_queue) {
            console.log('JOB QUEUE: adding |' + job_id + '|');
            job_queue.addBack({
              id: job_id,
              out_file: output_file,
            });
          }
        } else {
          console.log('WARN: Job |' + job_id + '| should have an output file!');
        }
      } else if (j_status == 'Running') {
        // Check job is running.
        if (job_id !== job_running) {
          console.log('WARN: Job |' + job_id + '| is not running!');
        }
      } else if (j_status == 'Done') {
        // If job is done, check if we have the associated output.
        if (!fs.existsSync(output_file)) {
          console.log('WARN: Job |' + job_id + '| is done and output file does not exist! It needs to be archived!');
        }
      } else if (j_status == 'Archived') {
        // If job is archived, we should not have the associated output.
        if (fs.existsSync(output_file)) {
          console.log('WARN: Job |' + job_id + '| is archived and output file exists! It should not be archived!');
        }
      } else {
        console.log('WARN: Job |' + job_id + '| has an error.');
      }

      // Populate return json with job data.
      j['job'] = {
        id: q_r.id,
        created: q_r.created,
        uname: q_r.username,
        status: j_status,
        output_file: output_file,
      };
    }
    else {
      // Got back multiple jobs. Should not be possible!
      j['err'] = true;
      j['err_msg'] = 'Could not find a job!';
    }
  } catch (error) {
    // Handle SQL error.
    j['err'] = true
    j['err_msg'] = error.toString();
  }

  // Write return data.
  res.statusCode = 200;
  res.json(j);
})


/*
  Endpoint to get a list of jobs associated with a user.
  @params: |user|: Username to 
  @output: json:  |err|: If there was any error
                  |err_msg|: Error message if there was an error.
                  |job|: List of job jsons associated with the user.
*/
app.get('/jobs_associated_to_user', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  const usr_name = req.query.user;

  // Return json.
  var j = {
    err: false,
    err_msg: '',
    jobs: [],
  };

  try {
    // Get data from SQL database.
    var q = `SELECT id FROM Users WHERE username='${usr_name}'`;
    var q_r = db_con.query(q);
    if (q_r.length == 1) {
      var q = `SELECT * FROM Jobs WHERE user=${q_r[0].id} ORDER BY created DESC;`;
      var q_r = db_con.query(q);

      q_r.forEach(q_e => {
        // Add each job associated with user.
        var output_file = q_e.output;
        if (output_file && !fs.existsSync(output_file)) {
          output_file = null;
        }

        j['jobs'].push({
          id: q_e.id,
          created: q_e.created,
          status: q_e.status,
          output_file: output_file,
        });
      });
    }
  } catch (error) {
    j['err'] = true
    j['err_msg'] = error.toString();
  }

  // Write return data.
  res.statusCode = 200;
  res.json(j);
})


/*
  Endpoint to delete a job.
  @params: |job_id|: Id of job to delete.
  @output: string: 'Success' | 'ERR: {msg}'
*/
app.get('/delete_job', function (req, res) {
  res.set({
    'Access-Control-Allow-Origin': '*',
  });
  const job_id = req.query.job_id;

  try {
    // Delete job from SQL database.
    var q = `DELETE FROM Jobs WHERE id='${job_id}'`;
    db_con.query(q);
    res.write('Success');
  } catch (error) {
    res.write('ERR: ' + error.toString());
  }

  // Write return data.
  res.statusCode = 200;
  res.end();
})


/*
  Endpoint to start a job. Adds to the job to the queue for execution.
  @params: |job_id|: Id of job to start.
           |user|: Username of user associated with job.
           |num_epochs|: Number of epochs to train for.
           |output_size|: Number of notes to generate for output song.
  @output: json:  |err|: If there was any error
                  |err_msg|: Error message if there was an error.
                  |status|: New status of job.
                  |output_file|: File where model output data is stored.
*/
app.get('/start_job', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });

  // Parse inputs and define variables. 
  const job_id = req.query.job_id;
  const usr_name = req.query.user;
  const out_file = `/home/dp/NetBeansProjects/musicAILocalSite/data_files/${usr_name}/${job_id}_log.log`;
  const job_status = 'Queued';

  // console.log(job_id);
  // console.log(usr_name);

  // Return value.
  var j = {
    err: false,
    err_msg: '',
    status: job_status,
    output_file: out_file,
  };

  try {
    // Update job status in SQL database.
    var q = `UPDATE Jobs SET status = "${job_status}", output = "${out_file}" WHERE id='${job_id}'`;
    db_con.query(q);

    // Add job to queue for execution at next available job.
    console.log('JOB QUEUE: adding |' + job_id + '|');
    job_queue.addBack({
      id: job_id,
      out_file: out_file,
      num_epochs: req.query.num_epochs,
      output_size: req.query.output_size,
    });
  } catch (error) {
    // Handle error.
    j['err'] = true
    j['err_msg'] = error.toString();
  }

  // Begin any job in queue if one isn't already running. 
  begin_next_job();

  // Write return data.
  res.statusCode = 200;
  res.json(j);
})


/*
  Endpoint to the get the status of a job. 
  Has support for live streaming of the latest results in the temporary output buffer.
  @params: |job_id|: Id of job.
           |job_out_file|: Output file of job to get status data from.
           |job_status|: Status of job.
           |read_file|: If data should be read from the |job_out_file|. If this is false, only data in the buffer will returned (ie: this allows for caller to get a live stream of the model output.)
  @output: json:  |err|: If there was any error
                  |err_msg|: Error message if there was an error.
                  |is_running|: if the job is still running
                  |status|: Current (updated) status of the job.
*/
app.get('/model_status', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  const job_id = req.query.job_id;
  const job_out_file = req.query.job_out_file;
  const job_status = req.query.job_status;
  const req_read_file = req.query.read_file == 'true';

  // Return json.
  var j = {
    err: false,
    err_msg: '',
    output_lines: [],
    is_running: false,
    status: null,
  };

  // Check if we should read from the |job_out_file| and/or the |model_out_buffer| buffer.
  should_read_file = false
  should_read_buffer = false
  if (job_status == 'Created') {
    j['err'] = true;
    j['err_msg'] = 'Job needs to be started before we can get status';
  } else if (job_status == 'Queued') {
    should_read_file = req_read_file;
    should_read_buffer = (job_id == job_running);
  } else if (job_status == 'Running') {
    should_read_file = req_read_file;
    should_read_buffer = (job_id == job_running);
  } else if (job_status == 'Done') {
    should_read_file = true;
  } else if (job_status == 'Archived') {
    j['err'] = true;
    j['err_msg'] = 'Job has been archived! Cannot get status.';
  } else {
    j['err'] = true;
    j['err_msg'] = 'Job had error! Cannot get status.';
  }

  // Read data from |job_out_file| file if necessary.
  if (should_read_file && job_out_file) {
    if (!fs.existsSync(job_out_file)) {
      console.log('WARN: file |' + job_out_file + '| does not exist!');
    } else {
      j['output_lines'] = fs.readFileSync(job_out_file).toString().split("\n");
    }
  }

  // Read data from |should_read_buffer| buffer if necessary.
  if (should_read_buffer) {
    // Ensure we get a lock on the buffer. Since we are emptying the buffer, we save data to the output file as well.
    while (model_out_file_lock);
    model_out_file_lock = true;
    if (job_id == job_running) {
      for (var i = 0; i < Math.min(model_out_buffer.size(), 20); i++) {
        append_to_model_file(model_out_buffer.peekFront());
        j['output_lines'].push(model_out_buffer.removeFront());
      }
      model_out_file_lock = false;
    }
  }

  // Begin any job in queue if one isn't already running. 
  begin_next_job();

  // Update status of job.
  j['is_running'] = (job_id == job_running);
  if (j['is_running']) {
    j['status'] = 'Running';
  } else if (job_id === job_last_done) {
    j['status'] = 'Done';
  } else if (job_status === 'Queued') {
    j['status'] = 'Queued';
  }
  if (job_status === 'Done' && j['status'] === null) {
    j['status'] = 'Done';
  }

  // Filter out some debug lines from TensorFlow
  // TODO(deev): Look into modifying the TensorFlow training code to not show this information. in the first place.
  j['output_lines'] = j['output_lines'].filter(l => !(l.includes('ETA: ')));

  // Write return data.
  res.statusCode = 200;
  res.json(j)
})


/*
  Endpoint to check if a specific output song (associate with a job) has been uploaded to the frontend website.
  @params: |fn|: Filename to check
  @output: string: 'Yes' | 'No'
*/
app.get('/is_uploaded', function (req, res) {
  res.set({
    'Access-Control-Allow-Origin': '*',
  });
  const fn = req.query.fn;

  // Check if file has been uploaded.
  if (ftpUploadedLst.includes(fn.substring(fn.lastIndexOf('/') + 1))) {
    res.write('Yes');
  } else {
    res.write('No');
  }

  // Write return data.
  res.statusCode = 200;
  res.end();
})


/*
  Endpoint to upload generated song to frontend website via FTP
  @params: |fn|: File name of generate song to upload.
  @output: json:  |err|: If there was any error
                  |err_msg|: Error message if there was an error.
*/
app.get('/upload_result', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });
  const fn = req.query.fn;
  const fn_remote = fn.substring(fn.lastIndexOf('/') + 1);

  // Return json.
  var j = {
    err: false,
    err_msg: '',
  };

  // Upload result via FTP.
  console.log('Uploading result: ' + fn);
  ftpClient.put(fn, ftpUploadDir + fn_remote, function (err, list) {
    if (err) {
      console.log('Uploading error: ' + err.toString());
      j['err'] = true;
      j['err_msg'] = err.toString();
    } else {
      console.log('Uploaded to: ' + fn_remote);
      ftpUploadedLst.push(fn_remote);
    }
  });

  // Write return data.
  res.statusCode = 200;
  res.json(j)
})


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Variables and functions to interface with twitter.
 *      TODO(deev): Look into moving the twitter stuff into a different node file as this one is getting too large to maintain efficiently.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


// Twitter API variables.
const twitter_consumer_config = {
  consumer_key: 'wal4PYmlr7Y6nUgiv1b5QduvQ',
  consumer_secret: 'ljAskgly0ft6YtjRgjGwEvZ7R01DeBPMiHKzPQZOdUVVPdjPaB',
}
const twitter_verify_link = 'https://api.twitter.com/oauth/authenticate?oauth_token=';
var twitter_oauth_token = null;

// Variables related to creating a post.
var twitter_post_msg = '';
var twitter_post_fn = '';
var twitter_post_media_id = '';
var twitter_acc_usr_id = '';

// Function to upload media file to twitter.
async function twitter_upload_media() {
  if (!twitter_post_fn) return;
  twitter_post_media_id = '';
  console.log('Uploading media!');

  // Look for and upload the file.
  var finder = findit('/home/dp/NetBeansProjects/musicAILocalSite/data_files/');
  finder.on('file', function (fn) {
    if (fn.endsWith(twitter_post_fn)) {
      console.log('Uploading file: ' + fn);

      const media_id_fn = '/home/dp/makeLoc/local_node_server/twitter_tmp/media_id.txt'

      // Run python script to upload data using twitter API.
      const spawn = require("child_process").spawn;
      const twitter_conversion_process = spawn('python3.9', ["/home/dp/makeLoc/local_node_server/twitter_media_upload.py",
        "--file_to_upload", fn,
        "--output_file", media_id_fn,
        "--twitter_acc_usr_id", twitter_acc_usr_id]);
      twitter_conversion_process.stdout.on('data', (data) => {
        console.log(data.toString())
      });
      twitter_conversion_process.stderr.on('data', (data) => {
        console.log(data.toString())
      });
      twitter_conversion_process.on('close', (code) => {
        twitter_post_media_id = fs.readFileSync(media_id_fn).toString()
        console.log('Media Upload done! Media id: ' + twitter_post_media_id);
      });
    }
  });
}


/** 
 * |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Endpoints for dealing with Twitter API and automatically creating posts
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


/*
 Endpoint to begin twitter posting process. This starts the verification process,
  @params: |post_msg|: Message for twitter post.
           |fn|: Media file to attach to the post.
  @output: string: Instructions on how to finish verification process.
*/
app.get('/twitter_verify_start', function (req, res) {
  res.set({
    'Access-Control-Allow-Origin': '*',
  });
  twitter_post_msg = req.query.post_msg;
  twitter_post_fn = req.query.fn;

  // Create client verification process. The process will redirect users to upload endpoint.
  const client = new twitter(twitter_consumer_config);
  client
    .getRequestToken("https://98.182.226.187:3000/upload_to_twitter")
    .then(res => {
      twitter_oauth_token = res.oauth_token;
      // console.log({
      //   reqTkn: res.oauth_token,
      //   reqTknSecret: res.oauth_token_secret,
      // });
    })
    .catch(console.error);

  // Write return data.
  res.write('To finish curl: /twitter_verify_complete');
  res.statusCode = 200;
  res.end();
})


/*
 Endpoint to complete the verification process. This redirects the user to the twitter login page.
  @params: None
  @output: None. Redirects user to twitter login page
*/
app.get('/twitter_verify_complete', function (req, res) {
  if (twitter_oauth_token) {
    // Redirect user to twitter login page.
    const tmp = twitter_oauth_token;
    twitter_oauth_token = null;
    res.redirect(twitter_verify_link + tmp);
  } else {
    // Error. Return user to main musicAi page. 
    console.log("No Token!")
    res.redirect('https://musicai.app');
  }
})


/*
 Endpoint to upload to twitter after verification process has been completed. 
 This endpoint is called automatically by the twitter link with the required data
  @params: Authentication tokens set automatically by twitter API. See: https://developer.twitter.com/en/docs/authentication/api-reference/access_token 
  @output: None. Redirects user to page in main website.
*/
app.get('/upload_to_twitter', function (req, res) {
  res.set({
    'Access-Control-Allow-Origin': '*',
  });

  // Create twitter client with necessary authentication data.
  const client = new twitter(twitter_consumer_config);
  client
    .getAccessToken({
      oauth_verifier: req.query.oauth_verifier,
      oauth_token: req.query.oauth_token
    }).then(res => {
      // console.log({
      //   accTkn: res.oauth_token,
      //   accTknSecret: res.oauth_token_secret,
      //   userId: res.user_id,
      //   screenName: res.screen_name
      // });
      twitter_acc_usr_id = res.user_id;
      twitter_config_user = Object.assign({}, twitter_consumer_config);
      twitter_config_user['access_token_key'] = res.oauth_token;
      twitter_config_user['access_token_secret'] = res.oauth_token_secret;
      // console.log(twitter_config_user);
      const client_user = new twitter(twitter_config_user);

      // Start twitter media upload.
      twitter_upload_media();

      // Wait a little while before creating post. This is because the twitter API takes sometime to process media files.
      // TODO(deev): Find a better way to do this. Maybe repeatedly queue twitter API until it has the media file?
      setTimeout(function () {
        console.log('Tweeting. Media id: ' + twitter_post_media_id)
        client_user.post('statuses/update',
          {
            status: twitter_post_msg,
            media_ids: twitter_post_media_id
          }).catch(console.error);
      }, 15000);
    }).catch(console.error);

  // Redirect user to musicAi website.
  res.redirect('https://musicai.app/twitter/twitter_complete.html');
})


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Variables and functions to interface with YouTube.
 *      TODO(deev): Look into moving the YouTube stuff into a different node file as this one is getting too large to maintain efficiently.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


// YouTube download variables.
var youtube_downloader_done = true;
var youtube_downloader_error = null;
const youtube_file_base = "/home/dp/makeLoc/local_node_server/youtube_tmp";
const youtube_file = "youtube_download.mp3";

// Downloader. See https://github.com/ytb2mp3/youtube-mp3-downloader.git for details on API.
var downloader = new youtube_downloader({
  "ffmpegPath": "/usr/bin/ffmpeg",        // Path the ffmpeg to do some conversions
  "outputPath": youtube_file_base,        // Where to store outputs
  "youtubeVideoQuality": "highestaudio",  // Desired video quality
  "queueParallelism": 2,                  // Download parallelism
  "progressTimeout": 2000,                // Interval in ms for the progress reports
  "allowWebm": false                      // Enable download from WebM sources
});

// Starts the download process.
function start_downloading(link) {
  if (!youtube_downloader_done) return;
  youtube_downloader_done = false;

  // Download video and save as MP3 file
  downloader.download(link, youtube_file);
  console.log("Starting Download!");

  // Process data from downloader.
  downloader.on("finished", function (err, data) {
    console.log("Finished Download!");
    youtube_downloader_done = true;
    youtube_downloader_error = null;
  });
  downloader.on("error", function (error) {
    console.log("Download error:" + error);
    youtube_downloader_done = true;
    youtube_downloader_error = error;
  });
  downloader.on("progress", function (progress) {
    console.log(JSON.stringify(progress));
  });
}


/** 
 * |||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Endpoints for dealing with YouTube and downloading songs from there.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


/*
  Endpoint to start song download from YouTube.
  @params: |link_url|: YouTube link to download. Only the video id (ex for https://www.youtube.com/watch?v=Vhd6Kc4TZls, you would pass only Vhd6Kc4TZls as the link)
  @output: string: Instructions to query status of download process | "Failure: {msg}"
*/
app.get('/start_youtube_download', function (req, res) {
  res.set({
    'Access-Control-Allow-Origin': '*',
  });

  // Start downloading. We only support downloading of one file at a time for now.
  if (youtube_downloader_done) {
    start_downloading(req.query.link_url);
    res.write("Query status: /get_youtube_download_status");
  } else {
    res.write("Failure: Try again later!");
  }

  // Write return data.
  res.statusCode = 200;
  res.end();
})


/*
  Endpoint to query status of YouTube download.
  @params: None
  @output: json:  |err|: If there was any error
                  |err_msg|: Error message if there was an error
                  |running|: If the download process is still running
                  |result_link|: Link to get downloaded song if the song is done downloading
*/
app.get('/get_youtube_download_status', function (req, res) {
  res.set({
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
  });

  // Return json.
  var j = {
    err: false,
    err_msg: '',
    running: false,
    result_link: '',
  };

  if (!youtube_downloader_done) {
    // Downloader is still running.
    j['running'] = true;
  } else if (youtube_downloader_error) {
    // There was an error.
    j['err'] = true;
    j['err_msg'] = youtube_downloader_error.toString();
  } else {
    // If downloader is done, upload the data the musicAi website via FTP and return a link to it.
    const remote_fn = ftpUploadDirYoutube + crypto.createHash('md5').update(String(Date.now())).digest('hex') + '.mp3';
    j['result_link'] = "https://musicai.app/" + remote_fn.substr(remote_fn.indexOf('public_html/') + 'public_html/'.length);
    console.log('Uploading youtube file to website!');
    ftpClient.put(youtube_file_base + "/" + youtube_file, remote_fn, function (err, list) {
      // if (err) {
      //   j['err'] = true;
      //   j['err_msg'] = err.toString();
      //   console.log('FTP youtube upload error: ' + err.toString());
      // }
      console.log('Done uploading youtube video over FTP!');
    });
  }

  // Write return data.
  res.statusCode = 200;
  res.json(j)
})


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Miscellaneous endpoints.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


/*
  Endpoint to prompt users to accept the SSL certificate. 
  This is necessary because our certificate is self signed and chrome does not always like that.
  @params: None
  @output: Redirects to main website.
*/
app.get('/accept_certificate', function (req, res) {
  res.redirect('https://musicai.app')
})


/** 
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 *      Start server.
 * ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
 */


httpsServer.listen(port)
