@echo off
cd /d %~dp0
set log_file=log.txt

:: Set the name for the conda environment
set conda_env_name=PranalGhildiyal_regressionInteractive

:: Activate the conda environment
call conda activate %conda_env_name%

(
    :: Check if the conda environment exists
    conda info --envs | findstr /i "\<%conda_env_name%\>" > nul
    if %errorlevel% neq 0 (
        :: Create a new conda environment with Python 3.8
        conda create --name %conda_env_name% python=3.8 -y >> %log_file% 2>&1
        :: Install dependencies from requirements.txt if not already installed
        conda activate %conda_env_name%
        pip install -r requirements.txt >> %log_file% 2>&1
    )

    :: Activate the conda environment again to ensure we are working within it
    conda activate %conda_env_name%

    :: Exit the conda environment when done (optional)
    conda deactivate >> %log_file% 2>&1

    :: Display a completion message
    echo Installation and execution complete. You can now close this window.
)

:: Deactivate the conda environment if it's still active
call conda deactivate >> %log_file% 2>&1