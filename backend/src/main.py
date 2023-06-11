from src.api import app
import subprocess

if __name__ == "__main__":
    subprocess.run("uvicorn main:app --reload", shell=True)
