from typing import List

from pydantic import BaseModel


class Prediction(BaseModel):
    """
    Represents a prediction.
    """

    product: str
    score: float
    reaction_image: str
    iupac_name: str
    synonyms: List[str]


class PredictionResult(BaseModel):
    """
    Represents the result of a prediction.
    """

    result: List[Prediction]


class PredictionRequest(BaseModel):
    """
    Represents the request for a prediction.
    """

    smiles: str