from e2b import Template

template = (
    Template()
    .from_image("e2bdev/code-interpreter:latest")
    .set_user("root")
    .set_workdir("/")
    # Install only essential packages - cache bust v2
    .run_cmd(
        "/usr/local/bin/python3.12 -m pip install --no-cache-dir statsmodels flaml==1.2.4 fastparquet pyarrow && echo 'build-v2'"
    )
    # Verify installation
    .run_cmd(
        "/usr/local/bin/python3.12 -c 'import statsmodels; import flaml; import pyarrow; from fastparquet import ParquetFile; from fastparquet import write; print(\"OK\")'"
    )
    .set_user("user")
    .set_workdir("/home/user")
    .set_start_cmd("sudo /root/.jupyter/start-up.sh", "sleep 20")
)
