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
1. Clone the Recommended Model: PSMNet (Pyramid Stereo Matching Network) Repository
`git clone https://github.com/JiaRenChang/PSMNet.git`

2. Download pretrained weights. Download the SceneFlow pretrained model from:
`https://drive.google.com/file/d/1pHWjmhKMG4ffCrpcsp_MTXMJXhgl3kF9/view`

3. Adjust for your GPU:
The GTX 960 has 4GB VRAM, so you may need to:

Reduce batch size (use 1)

Process smaller image patches if needed

Use maxdisp=96 instead of 192 for lower resolution

3. Prepare Demo Images
Put your stereo pair into the data\stereo\ directory or a subfolder. The code expects image pairs like im0.png and im1.png.

If needed, rename your stereo images accordingly.

5. Build the Docker Image and connect to it
See "set up docker..." above

