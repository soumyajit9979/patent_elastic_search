import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

engine = Elasticsearch(os.getenv("db_url"))


def get_db():
    return engine