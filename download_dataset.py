from dotenv import load_dotenv
from roboflow import Roboflow
import os

load_dotenv()

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("workspace-epi").project("hard-hat-universe-0dy7t-tbkpp")
version = project.version(1)
dataset = version.download("yolov8", location="dataset")
