<!DOCTYPE html>
<?php
$local_dir = '/home/dp/NetBeansProjects/musicAILocalSite/data_files/';
$supported_extensions = array(".mid", '.midi');
?>

<?php

function startsWith($str, $search) {
    return substr_compare($str, $search, 0, strlen($search)) === 0;
}

function endsWith($str, $search) {
    return substr_compare($str, $search, -strlen($search)) === 0;
}
?>

<html>
    <head>
        <title> Training File Download </title>
    </head>
    <body>
        <?php
        # var_dump($_FILES['file']);

        $n_files = count($_FILES['file']['name']);
        if (!$n_files) {
            die('No files given!');
        }


        # Santiy check file sizes and extensions.
        for ($i = 0; $i < count($_FILES['file']['name']); $i++) {
            $f_n = $_FILES['file']['name'][$i];
            $f_s = $_FILES['file']['size'][$i];

            if ($f_s > 1024 * 1024) {
                die('ERR: file {' . $f_n . '} is too large {' . $f_s . ' bytes}');
            }

            $f_extensions = array_map(function($s) use($f_n) {
                return endsWith($f_n, $s);
            }, $supported_extensions);
            if (count(array_intersect($f_extensions, array(True))) === 0) {
                die('ERR: file {' . $f_n . '} is has an unsupported extension!');
            }
        }

        # Delete old files.
        $files = glob($local_dir . '*');
        foreach ($files as $f_n) {
            if (is_file($f_n)) {
                unlink($f_n);
            }
        }

        # Copy files. 
        for ($i = 0; $i < count($_FILES['file']['name']); $i++) {
            $f_n = $_FILES['file']['name'][$i];
            $f_tmp = $_FILES['file']['tmp_name'][$i];
            $f_n_new = 'training_' . $i . '_' . basename($f_n);

            if (move_uploaded_file($f_tmp, $local_dir . $f_n_new)) {
                echo 'Success: File {' . $f_n . '} has been uploaded as {' . $f_n_new . '}';
                echo '<br>';
            } else {
                die('ERR: could not upload file {' . $f_n . '}');
            }
        }
        ?>
        <br>
        <a href="/status.html">View training status</a>
    </body>
</html>