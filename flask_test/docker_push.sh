#!/bin/bash 

echo "image name: $1" 
echo "repo name: $2"

echo "=== Make remote image ==="
docker tag $1 $2/$1

echo "=== Push the image ==="
docker push $2/$1