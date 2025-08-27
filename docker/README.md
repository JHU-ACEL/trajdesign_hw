# Docker 

### Build the Docker Container
**Linux/Mac:**

Use the `docker_build.sh` script to build the Docker image:
```bash
./docker/docker_build.sh docker/Dockerfile trajdesign:v1
```

**Arguments:**
- `dockerfile`: Path to the Dockerfile (e.g., `docker/Dockerfile`)
- `tag_name`: Docker image tag (e.g., `trajdesign:v1`)

The build script automatically:
- Sets the user ID and group ID to match your current user
- Sets the username to your current username
- Builds the image with the specified tag

**Windows:**

Windows require a few additional steps to build the docker image. Git converts
LF endings to CRLF endings automatically when pulling .sh scripts on a windows system to maintain standard 
as linux/mac uses LF line endings while windows uses CRLF endings. While this can be configured to be turned off in git,
It is not recommended. Hence, we'd manually convert the shell scripts back to LF standard using dos2unix command in wsl.

**Option A:**

Manually convert the .sh files in unix format

1 - Open wsl and first intall dos2unix
```bash
apt install dos2unix
```
2 - Navigate to the "docker" folder inside the directory where you cloned the repository, e.g:
```bash
 cd /mnt/c/github/trajdesign_hw1/docker
```
3 - Run the dos2unix command to convert the .sh scripts
```bash
 dos2unix docker_build.sh docker_run.sh
```
4 - Navigate inside the setup folder and run the following command
```bash
 cd setup
 dos2unix  install_acados.sh install_tera_renderer.sh
```
**Option B:**

Turn auto crlf off in Git bash. Delete the repository if you've already cloned. Open Git Bash and enter the following command. Clone the repository after executing this
command in your preferred local directory
```bash
 cd setup
 git config --global core.autocrlf false
```
Go to the parent repository directory (trajdesign_hw1) and run the following command to build the docker image
```bash
./docker/docker_build.sh docker/Dockerfile trajdesign:v1
```
**Arguments:**
- `dockerfile`: Path to the Dockerfile (e.g., `docker/Dockerfile`)
- `tag_name`: Docker image tag (e.g., `trajdesign:v1`)

The build script automatically:
- Sets the user ID and group ID to match your current user
- Sets the username to your current username
- Builds the image with the specified tag

### Run the Docker Container
Use the `docker_run.sh` script to run the Docker container:

**Linux/Mac:**
```bash
./docker/docker_run.sh trajdesign:v1
```
**Windows:**
```bash
### On Windows start with a port as an argument as default, e.g:
./docker/docker_run.sh trajdesign:v1 8888
```

**Arguments:**
- `tag_name`: The Docker image tag to run (e.g., `trajdesign:v1`)
- `ssh_port` (optional): The port forwarding between host machine and Docker container (e.g., `8888`)

### Run marimo from inside the container

```bash
### Run marimo (from inside the container):
marimo edit --headless --host 0.0.0.0 --port=$PORT
```

Now type in `localhost:{$PORT}` into your browser and copy/paste the access token to start using the notebook.


