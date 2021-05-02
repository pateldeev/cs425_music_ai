# Python script to upload a media file to twitter using the twitter API.
# Converts a song file to a video and saves the twitter |media_id|.
# Twitter requires splitting up the media file into multiple parts.
# Relevant API Docs: https://developer.twitter.com/en/docs/twitter-api/v1/media/upload-media/api-reference/post-media-upload
# Requires |twurl| to be setup and linked to twitter app doing the uploading (https://github.com/twitter/twurl)

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

# Commandline arguments.
parser = argparse.ArgumentParser(description="Upload file to twitter using API's media upload")
parser.add_argument("--file_to_upload", type=str, default=None, help="File to upload")
parser.add_argument("--twitter_acc_usr_id", type=str, default=None, help="Twitter user id")
parser.add_argument(
    "--tmp_dir", type=str, default="/home/dp/makeLoc/local_node_server/twitter_tmp/", help="Temporary directory"
)
parser.add_argument(
    "--output_file",
    type=str,
    default="/home/dp/makeLoc/local_node_server/twitter_tmp/media_id.txt",
    help="file to output media id",
)
args = parser.parse_args()

if __name__ == "__main__":
    # Validate inputs.
    assert args.file_to_upload, "Need file to upload!"
    assert args.twitter_acc_usr_id, "Need user id!"
    assert args.file_to_upload.endswith(".mp3"), "Need mp3 file!"

    # Create or clear temporary directory if necessary.
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)
    else:
        for f in glob.glob("{}*".format(args.tmp_dir)):
            os.remove(f)

    # Copy input file.
    fn_local = "media_cpy.mp3"
    fn = "{}{}".format(args.tmp_dir, fn_local)
    shutil.copyfile(args.file_to_upload, fn)

    # Convert to video file.
    fn_video = "{}media.mp4".format(args.tmp_dir)
    cmd_to_run = "ffmpeg -loop 1 -i /home/dp/Downloads/musicai.jpg -i {}{} -vf \"scale='min(1280,iw)':-2,format=yuv420p\" -c:v libx264 -preset medium -profile:v main -c:a aac -shortest -movflags +faststart {}".format(
        args.tmp_dir, fn_local, fn_video
    )
    print("Converting to mp4: ", cmd_to_run)
    subprocess.run(
        cmd_to_run,
        capture_output=True,
        check=True,
        shell=True,
    )

    # Get file size.
    cmd_to_run = "stat --format='%s' {}".format(fn_video)
    p = subprocess.run(
        cmd_to_run,
        capture_output=True,
        check=True,
        shell=True,
    )
    fn_video_size = int(p.stdout)
    print("File size: ", fn_video_size)

    # Split file into pieces compatable with twitter API.
    split_fmt = "{}split_".format(args.tmp_dir)
    cmd_to_run = "split -b 5m {} {}".format(fn_video, split_fmt)
    subprocess.run(
        cmd_to_run,
        capture_output=True,
        check=True,
        shell=True,
    )

    # Initialize upload.
    cmd_to_run = "twurl -X POST -H upload.twitter.com '/1.1/media/upload.json?additional_owners={}' -d 'command=INIT&media_type=video/mp4&media_category=amplify_video&total_bytes={}'".format(
        args.twitter_acc_usr_id, str(fn_video_size)
    )
    print("Initalizing: ", cmd_to_run)
    p = subprocess.run(cmd_to_run, capture_output=True, check=True, shell=True, text=True)
    p_out = str(p.stdout)
    print("|||||||||")
    print(p_out)
    print("|||||||||")
    media_id = json.loads(p_out)["media_id"]
    print("Media id: ", media_id)

    # Upload each split.
    for split_i, split_fn in enumerate(sorted(glob.glob("{}*".format(split_fmt)))):
        cmd_to_run = "twurl -X POST -H upload.twitter.com '/1.1/media/upload.json' -d 'command=APPEND&media_id={}&segment_index={}' --file {} --file-field 'media'".format(
            media_id, split_i, split_fn
        )
        print("Uploading split", split_i, ":", cmd_to_run)
        subprocess.run(cmd_to_run, capture_output=True, check=True, shell=True)

    # Finish upload.
    cmd_to_run = "twurl -X POST -H upload.twitter.com '/1.1/media/upload.json' -d 'command=FINALIZE&media_id={}'".format(
        media_id
    )
    print("Finishing Upload", cmd_to_run)
    subprocess.run(cmd_to_run, capture_output=True, check=True, shell=True)

    # Save media id.
    with open(args.output_file, "w") as f:
        f.write(str(media_id))
    print("Done!")
