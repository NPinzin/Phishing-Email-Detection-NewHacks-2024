from pydantic import BaseModel

class EmailContent(BaseModel):
    content: str
