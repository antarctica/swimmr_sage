# Create a docker image that contains Python.
# The image can be used to run the SPIDER GGF forecast application.

# Use a specific version of Python
FROM python:3.8.5

# Set working directory inside the Docker build
WORKDIR /root/run_environment/

#Clone the github repository to the working directory in the Docker environment
#RUN git clone https://github.com/sagebas/sagebas .

# Take all the files located in the current directory and copy them into the Docker image
COPY . .

# Install the Python packages we'll use
RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r ./requirements.txt

# Alter permissions
RUN chmod +x ./GGF_RTF.py
RUN chmod +x ./GGF_RTFH.py
#RUN chmod +x ./hello_world.py

# I believe this updates $PATH to include /usr/bin/python3, which is required before running Python scripts
ENTRYPOINT [ "python3" ]
