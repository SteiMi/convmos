# docker build torch-gdal-docker -t lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal
# docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal
# docker tag lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal:1.2.0
# docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/torch-gdal:1.2.0

VERSION=1.4.10

docker build . -t lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next
# Push image to our docker registry
# docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next
docker tag lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next:${VERSION}
docker push lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next:${VERSION}
docker tag lsx-harbor.informatik.uni-wuerzburg.de/steininger-statistical-downscaling/sd-next:${VERSION} steimi/sd-next:${VERSION}
docker push steimi/sd-next:${VERSION}
