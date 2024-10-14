from pydantic import BaseModel

class Roles(BaseModel):
    role : str
    role_name : str
