FROM lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal:1.2.0

COPY requirements.txt .

# Install other requirements
RUN pip install -r requirements.txt

COPY models ./models
COPY remo_eobs_land_mask.npy ./
COPY data ./data

RUN chmod -R 777 ./

COPY *.py ./
RUN chmod 777 *.py

CMD [ "python", "run.py" ]
