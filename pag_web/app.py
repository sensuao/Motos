from flask import Flask, flash, render_template, request, redirect, url_for, send_file, stream_with_context, jsonify
# from flask_login import LoginManager, login_required
import boto3
import os
import docker
# from botocore.exceptions import ClientError
# from werkzeug.utils import secure_filename
# import os
# import subprocess
import paramiko
import time
from tqdm import tqdm

app = Flask(__name__, template_folder='templates', static_folder='static')
# app.config.from_object('config.Config')
# login_manager = LoginManager(app)
s3 = boto3.resource('s3')
# bucket = s3.Bucket("cbir-motos")
bucket_name = "cbir-motos"
s3_client = boto3.client('s3')
# ec2_client
ec2 = boto3.client('ec2')
id_instance = "i-08c1a9e1bfd6d961e"
ip_elastic = "54.74.199.9"


@app.route('/')
# @login_required
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ["zip"]


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):

            # Obtener el tamaño del archivo
            file_size = os.fstat(file.fileno()).st_size

            # Crear la barra de progreso
            progress_bar = tqdm(total=file_size, unit='B', unit_scale=True)

            # Definir función de actualización de la barra de progreso
            def progress_callback(bytes_uploaded):
                progress_bar.update(bytes_uploaded)

            # Subir el archivo a S3
            s3_client.upload_fileobj(
                file,
                bucket_name,
                "Data/" + file.filename,
                Callback=progress_callback
            )

            # Cerrar la barra de progreso
            progress_bar.close()

            return redirect(url_for('index'))
    return render_template('upload.html')


@app.route('/start', methods=['GET','POST'])
def start():
    response = ec2.start_instances(InstanceIds=[id_instance])

    # Esperar hasta que la instancia esté en ejecución
    waiter = ec2.get_waiter('instance_running')
    waiter.wait(InstanceIds=[id_instance])

    # Esperar hasta que la instancia esté en estado 'ok'
    ec2.get_waiter('instance_status_ok').wait(InstanceIds=[id_instance])

    # Conectar a la instancia mediante SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip_elastic, username='ec2-user', pkey=paramiko.RSAKey.from_private_key_file("claves.pem"))

    # Ejecuta el contenedor
    ssh.exec_command("docker run -v ~/.aws:/root/.aws processing_docker")

    return redirect(url_for("running"))


@app.route('/running', methods=['GET', 'POST'])
def running():
    # Wait for the container to finish executing
    # Conectar a la instancia mediante SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip_elastic, username='ec2-user', pkey=paramiko.RSAKey.from_private_key_file("claves.pem"))
    # Renderizar la plantilla waiting.html

    @stream_with_context
    def generate():
        yield render_template('waiting.html')
        while True:
            # Check the container's status
            _, stdout, _ = ssh.exec_command("docker ps -a --format '{{.Status}}' --filter ancestor=processing_docker")
            container_status = stdout.read().decode('utf-8').strip()

            if "Exited" in container_status:
                yield render_template('finished.html')
                ssh.exec_command("docker rm - f $(docker ps -a -q)")
                # Close the SSH connection
                ssh.close()
                return redirect(url_for("download"))

            time.sleep(3)  # Wait for 1 second before checking again

    return app.response_class(generate(), mimetype='text/html')


@app.route('/download', methods=['GET', 'POST'])
def download():
    # Descarga un archivo desde S3
    s3_client.download_file(bucket_name, "Results/move_images.py", "move_images.py")
    # Aquí deberías agregar el código necesario para descargar el archivo
    file_path = 'move_images.py'
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
