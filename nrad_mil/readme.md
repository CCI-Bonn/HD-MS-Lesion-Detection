# Dev usage
Developement inside docker container:
1. Bind repository folder to container:
    docker run [...]
        -v /path/to/repo:/nrad_torchlib
        [...]
2. Inside Dockerfile add:
    COPY . /nrad_torchlib
    RUN cd /nrad_torchlib && \
        pip install --editable .

After this nrad_torchlib is available as an installed python package.
'--editable' switch provides dev functionality. 
Changes in repo code take effect immediatly (after interpreter / python kernel restart ofc).
Folder inside container can be chosen arbitrarily (doesn't have to be /nrad_torchlib).
