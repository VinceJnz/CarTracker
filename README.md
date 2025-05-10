# Car tracker

Take a video feed and extract vehicle number plates


## Set up docker etc to run the current build

1. Start the container in detached mode
```bash
docker compose up -d
```

2. This will display a list of running containers along with their container IDs and names.
```bash
docker ps
```

3. To attach to the container's terminal, use the following command:
```bash
docker exec -it <container_name_or_id> /bin/bash
```



## Stereo vision

Set up an algorythm that looks at an area of an image an tries to match it with an area of a second image taken from a slightly different location.


Step-by-Step: Run AnyNet in Docker (GPU-accelerated, PyTorch-only)
1. Clone the AnyNet Repository
`git clone https://github.com/mileyan/AnyNet.git`

2. Create Dockerfile
```bash
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system packages
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-opencv git ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy code into container
COPY . .

# Install Python packages
RUN pip install --upgrade pip
RUN pip install torch torchvision opencv-python matplotlib tqdm

# Entry point — you can override this when running
CMD ["python", "demo.py"]
```

3. Prepare Demo Images
Put your stereo pair into the AnyNet/ directory or a subfolder. The code expects image pairs like im0.png and im1.png.

If needed, rename your stereo images accordingly.

4. Download Pretrained Model (on host)
`wget https://www.dropbox.com/s/5p8k5z9hw0qf4p4/anynet_final.tar -O checkpoints/anynet_final.tar`

5. Build the Docker Image
`docker build -t anynet-pytorch .`

6. Run the Docker Container with GPU Access
```bash
docker run --rm --gpus all \
  -v $(pwd):/app \
  anynet-pytorch \
  python demo.py --datapath ./ --loadmodel checkpoints/anynet_final.tar
```

7. Optional Improvements
* You can modify demo.py to output disparity images to a file rather than display them.
* You can mount an external folder for input/output images using -v.

