FROM python:3.9.13
RUN mkdir /app
WORKDIR app

copy . .
RUN pip install -r requirements.txt