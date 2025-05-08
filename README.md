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



