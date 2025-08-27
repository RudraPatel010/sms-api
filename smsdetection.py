from pydantic import BaseModel

class SmsDetection(BaseModel):
    message: str