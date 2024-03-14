import boto3
import os

s3 = boto3.resource('s3') # assumes credentials & configuration are handled outside python in .aws directory or environment variables


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            print("Downloading from directory: " + obj.key)
            continue
        bucket.download_file(obj.key, target)


def upload_s3_folder(bucket_name, local_folder, s3_dir):
    """
    Upload the contents of a folder directory to S3 bucket.
    :param bucket_name: the name of the s3 bucket
    :param local_folder: path of the folder in the local file system
    :param s3_dir: the folder path in the s3 bucket
    :return:
    """

    for file in os.listdir(local_folder):
        if file.endswith(".JPG"):
            s3.Bucket(bucket_name).upload_file(os.path.join(local_folder, file), s3_dir + "/" + file)
        else:
            folder = file
            boto3.client('s3').put_object(Bucket=bucket_name, Key=(s3_dir + "/" + folder + "/"))
            upload_s3_folder(bucket_name, local_folder + "/" + folder, s3_dir + "/" + folder)

