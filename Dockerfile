FROM nvidia/cuda:12.4.1-runtime-ubi9

# Install required libraries and dependencies for Streamlit
RUN yum install -y \
    python \
    python-pip \
 && yum clean all

ENV device_type=cuda

# Copy the requirements.txt file into the container
COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH /usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib/:/usr/local/lib/python3.9/site-packages/nvidia/cuda_cupti/lib/:$LD_LIBRARY_PATH

 # Set the working directory inside the container
WORKDIR /app
RUN mkdir SOURCE_DOCUMENTS

COPY . .
RUN chmod +x /app/redhatai.py

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the entry point for the container
ENTRYPOINT ["streamlit", "run", "redhatai.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
