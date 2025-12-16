from pydantic import BaseModel

class CreditScoringInput(BaseModel):
    RFMS_Score: float
    RecencyScore: float
    PricingStrategy: str
    ProductCategory: str