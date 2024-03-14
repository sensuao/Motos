FROM ubuntu:20.04
COPY . .
RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
CMD ["python3", "main_source_color.py"]