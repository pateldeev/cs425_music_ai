<!DOCTYPE html>
<?php
$web_server = 'freehosting.com';
$web_user = 'riyadhhossain736@gmail.com';
$web_server = 'musicai42!';

$ftp_server = 'cpanel.freehosting.com';
$ftp_user = 'upload@musicai.app';
$ftp_pw = 'upload24!';

$remote_dir = 'public_html/results/';
?>


<html>
    <head>
        <title> Result Upload </title>
    </head>
    <body>
        <?php
        $f_n_local = $_POST['fname'];
        $f_n_remote = $remote_dir . 'generated_song.mp3';

        # Connect to FTP server
        $ftp = ftp_connect($ftp_server) or die('Failed to connect to FTP server!');
        ftp_login($ftp, $ftp_user, $ftp_pw) or die('Failed to login to FTP account!');

        # Upload file
        $ret = ftp_nb_put($ftp, $f_n_remote, $f_n_local);
        while (FTP_MOREDATA == $ret) {
            
        }

        echo 'Song uploaded!';
        echo '<br>';
        ?>
        <br>
        <a href="http://musicai.app">Main page</a>
    </body>
</html>