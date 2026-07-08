
from pydantic import BaseModel
from typing import List

# FEATURE INSIGHTS FOR LLM

class FeatureInsight(BaseModel):
                feature: str
                importance: float
                interpretation: str
                business_implication: str
                confidence_note: str 

# MAIN REPORT

class BusinessReport(BaseModel):
                executive_summary: str
                key_drivers: List[FeatureInsight]
                recommendations: List[str]
                quick_wins: List[str]
                risks: List[str]
                final_summary: str  