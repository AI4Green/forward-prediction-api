from typing import List

from pydantic import BaseModel


class PubChemName(BaseModel):
    iupac_name: str
    synonyms: List[str]
