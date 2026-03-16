FROM python:3.9

RUN useradd toxpro

WORKDIR /home/toxpro

COPY requirements.txt requirements.txt
RUN python -m venv venv
RUN venv/bin/pip install -r requirements.txt

# netcat is a program
# necessary for troubleshooting
# the networking
RUN apt-get update && apt-get install -y netcat-traditional


COPY app app
#RUN pip install pyopenssl
RUN mkdir logs
RUN mkdir data
RUN mkdir instance # this is necessary for digital ocean

COPY boot.sh ./
RUN chmod +x boot.sh

COPY boot_worker.sh ./
RUN chmod +x boot_worker.sh

COPY boot_dashboard.sh ./
RUN chmod +x boot_dashboard.sh

RUN apt-get install libxrender1
ENV FLASK_APP app.py

RUN chown -R toxpro:toxpro ./
USER toxpro

EXPOSE 5000

#ENTRYPOINT ["./boot.sh"]