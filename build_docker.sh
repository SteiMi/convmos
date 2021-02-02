# docker build torch-gdal-docker -t lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal
# docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal
# docker tag lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal:1.0.0
# docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal:1.0.0

docker build . -t lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next
# Push image to our docker registry
# docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next
docker tag lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next:1.1.0
docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next:1.1.0
