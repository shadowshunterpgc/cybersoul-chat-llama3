from pydantic import BaseModel
from typing import Optional


class Dialogo(BaseModel):
    request: str
    response: Optional[str] = None
