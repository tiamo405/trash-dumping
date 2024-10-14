from typing import Optional
from pydantic import BaseModel, Field
import secrets
from fastapi import Form
class Auth(BaseModel):
    name: Optional[str] = None
    email: str
    password: str
    password_reset_token: str = Field(default_factory=lambda: secrets.token_hex(16))
    role: str = Field(default="USER")

    @classmethod
    def as_form(
        cls,
        name: Optional[str] = Form(None),
        email: str = Form(...),
        password: str = Form(...),
    ) -> 'Auth':
        return cls(
            name=name,
            email=email,
            password=password,
        )
    
class Customer(BaseModel):
    name: str
    email: str
    password: str
    role: str 
    

    