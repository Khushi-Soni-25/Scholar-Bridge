import os
from motor.motor_asyncio import AsyncIOMotorClient

_client = None
_db = None

def get_db():
    global _client, _db
    uri = os.getenv("MONGODB_URI")
    name = os.getenv("MONGODB_DB_NAME", "scholarbridge")
    if not uri:
        return None
    if _db is None:
        _client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        _db = _client[name]
    return _db
