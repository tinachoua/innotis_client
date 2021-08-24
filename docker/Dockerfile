FROM continuumio/miniconda3

RUN apt update
RUN apt install libgl1-mesa-glx -y
RUN apt install git -y

# Prepare InnoTIS 
RUN git clone https://github.com/MaxChangInnodisk/InnoTIS.git
WORKDIR "/InnoTIS"

# Create the conda environment:
RUN conda env create -f docker/environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "innotis", "/bin/bash", "-c"]

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "innotis", "python", "app.py"]
