"""Import the supplied scholarship CSV knowledge base into MongoDB Atlas.
Run after setting MONGODB_URI and MONGODB_DB_NAME in .env.
"""
import os, csv, asyncio
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name('.env'))

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
MAP={
 'Gujarat_Scholarships_Master.csv':'scholarships_master',
 'Eligibility_Rules.csv':'eligibility_rules',
 'Benefit_Details.csv':'benefit_details',
 'Document_Requirements.csv':'document_requirements',
 'Gujarat_Portals_Reference.csv':'portals_reference',
}
async def main():
 uri=os.getenv('MONGODB_URI'); dbname=os.getenv('MONGODB_DB_NAME','scholarbridge')
 if not uri: raise SystemExit('MONGODB_URI is missing')
 client=AsyncIOMotorClient(uri); db=client[dbname]
 for file,col in MAP.items():
  path=DATA/file
  if not path.exists(): continue
  with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
  if rows:
   await db[col].delete_many({})
   await db[col].insert_many(rows)
   print(f'{col}: {len(rows)} documents')
 await db.users.create_index('email',unique=True)
 await db.profiles.create_index('user_id',unique=True)
 await db.saved_scholarships.create_index([('user_id',1),('scholarship_id',1)],unique=True)
 print('MongoDB seed complete.')
 await client.close()
if __name__=='__main__': asyncio.run(main())
