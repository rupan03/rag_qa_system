# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install all dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Tell Hugging Face Spaces which port the app will run on
EXPOSE 7860

# The command to start the Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]