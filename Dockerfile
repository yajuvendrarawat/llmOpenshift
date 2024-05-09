FROM nvidia/cuda:12.4.1-runtime-ubi9

# Install required libraries and dependencies for Streamlit
RUN yum install -y \
    python3==3.9.18 \
    python3-pip==3.9.18 \
 && yum clean all

 # Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV LD_LIBRARY_PATH /usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib/:/usr/local/lib/python3.9/site-packages/nvidia/cuda_cupti/lib/:$LD_LIBRARY_PATH


