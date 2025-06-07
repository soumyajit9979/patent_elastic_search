import os
import json
from dotenv import load_dotenv

load_dotenv()

debug: bool = os.getenv("DEBUG")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
X_API_KEY = os.getenv("X_API_KEY")
db_url: str = os.getenv("db_url")
