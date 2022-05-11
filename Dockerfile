FROM python:3.6
MAINTAINER ganeshbarma barmaganesh@gmail.com
WORKDIR ./
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["gunicorn", "live_detect_mask:app"]
 
