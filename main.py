from fastapi import FastAPI 
app = FastAPI()
import os
odoo_api_key = os.getenv('sahayak-ai-odoo')
