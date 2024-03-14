import logging
import os
import time
import shutil
import boto3

s3_data_path = ""
data_zip_name = "caca.zip"
machine_data_path = "Data_original"
machine_rs_data_path = machine_data_path + "/" + data_zip_name

# S3 parameters
s3 = boto3.resource('s3')
bucket = s3.Bucket("cbir-motos")

# LOGGINGS
loggigns_file_name = 'loggings_test.log'
# DOWNLOAD LOG FILE
bucket.download_file('Logs/' + loggigns_file_name, loggigns_file_name)
# create a logger
logger = logging.getLogger('logger')
# set logger level
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler(loggigns_file_name)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_data_from_s3(bucket, data_zip_name, machine_data_path):
    # DOWNLOAD ZIP FROM S3
    logger.info("Downloading zip file:" + data_zip_name)
    bucket.download_file(data_zip_name + ".zip",
                         machine_data_path + "/" + data_zip_name + ".zip")
    logger.info("Zip file successfully downloaded")


def unpack_and_delete_zip_file(data_zip_name, machine_data_path):
    # UNPACK ZIP FILE
    logger.info("Unpacking zip file")
    shutil.unpack_archive(machine_data_path + '/' + data_zip_name + ".zip", machine_data_path, "zip")
    logger.info("Zip file successfully unpacked")
    os.remove(machine_data_path + '/' + data_zip_name + ".zip")


def set_state(state, state_filename="state.txt"):
    if state == "start":
        idx = 0
    if state == "finished":
        idx = 2
    bucket.download_file(state_filename, "state.txt")
    with open('state.txt', 'r') as file:
        data = file.read().rstrip()
    data = list(data)
    data[idx] = "1"
    data = "".join(data)
    with open('state.txt', 'w') as file:
        file.write(data)


def main():

    os.mkdir(machine_data_path)

    data_zip_name = "caca"

    get_data_from_s3(bucket, data_zip_name, machine_data_path)

    unpack_and_delete_zip_file(data_zip_name, machine_data_path)

    with open(os.path.join(machine_data_path, 'data.txt'), 'r') as file:
        data = file.read().rstrip()

    # RESULT
    print(data)
    logger.info(data)

    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info("DONE WITH " + data_zip_name)
    logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    # LOG FINALE
    logger.info("All finished")

    # UPLOAD LOG FILE
    bucket.upload_file(loggigns_file_name, 'Logs/' + loggigns_file_name)


if __name__ == "__main__":
    main()
