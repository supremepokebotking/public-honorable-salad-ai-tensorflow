# We simply inherit the Python 3 image. This image does
# not particularly care what OS runs underneath
FROM python:3.8
ENV PYTHONUNBUFFERED 1
# Set an environment variable with the directory
# where we'll be running the app
ENV APP /app
# Create the directory and instruct Docker to operate
# from there from now on
RUN mkdir $APP
WORKDIR $APP
# Copy the requirements file in order to install
# Python dependencies
COPY requirements.txt .
# Install Python dependencies
RUN pip install -r requirements.txt
# We copy the rest of the codebase into the image
COPY . .

# Expose the port uWSGI will listen on
EXPOSE 9897

ENTRYPOINT [ "python3" ]
CMD [ "app.py" ]
