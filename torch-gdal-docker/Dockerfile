FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y \
    software-properties-common

RUN add-apt-repository ppa:ubuntugis/ubuntugis-unstable

# Existing binutils causes a dependency conflict, correct version will be installed when GDAL gets intalled
RUN apt-get remove -y binutils

# Avoid tzdata wanting a prompt
RUN ln -fs /usr/share/zoneinfo/Europe/Berlin /etc/localtime

# Install GDAL dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y \
    libgdal-dev \
    g++ \
    python-tk \
    libsm6 \
    libxext6 \
    libxrender-dev

# Update C env vars so compiler can find gdal
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install correct GDAL version depending on the one installed in the container
RUN pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

# Install Latex to have nice text in plots based on this image
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -q && apt-get install -qy \
    texlive-full \
    python-pygments gnuplot \
    make git \
    && rm -rf /var/lib/apt/lists/*


CMD [ "echo", "You should add your code and change this command!" ]
