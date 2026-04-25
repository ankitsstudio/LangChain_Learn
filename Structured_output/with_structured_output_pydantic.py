from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class Response(BaseModel):
    key_themes: List[str] = Field(description="The key themes or topics mentioned in the text")
    summary: str = Field(description="A brief summary of the text")
    sentiment: Literal["pos", "neg", "neutral"] = Field(..., description="The sentiment of the text")
    metadata: Optional[dict] = Field(None, description="Additional metadata about the text")
    pros: Optional[List[str]] = Field(None, description="A list of positive aspects mentioned in the text")
    cons: Optional[List[str]] = Field(None, description="A list of negative aspects mentioned in the text")

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    deployment_name="gpt-4o-mini",
    api_version="2025-01-01-preview"
)

structured_model = model.with_structured_output(Response)
response = structured_model.invoke("""The Smart Fitness Band X200 tries to position itself as a budget-friendly fitness tracker, but it falls short in several important areas, making it hard to recommend.

Pros:
Affordable Price: One of the few redeeming qualities is its low cost, making it accessible for beginners.
Lightweight Design: Comfortable to wear for long periods without irritation.
Basic Tracking Features: It does cover essentials like step count and heart rate monitoring.
                                   
Cons:
Inaccurate Data: The fitness tracking is inconsistent. Step counts and heart rate readings often feel unreliable, which defeats the main purpose of a fitness band.
Poor Battery Performance: Despite claims, the battery drains much faster than expected, especially with continuous tracking enabled.
Cheap Build Quality: The materials feel flimsy, and the strap doesn’t seem durable for long-term use.
Limited App Functionality: The companion app is buggy, slow, and lacks meaningful insights or analytics.
Display Issues: The screen visibility in sunlight is poor, making it inconvenient for outdoor use.
Final Verdict:

While the Smart Fitness Band X200 might look appealing due to its price, the compromises in accuracy, build quality, and overall user experience make it a disappointing purchase. Spending a bit more on a reliable alternative would likely be a better investment.""")
print(f"Key Themes: {response.key_themes}")
print(f"Summary: {response.summary}")
print(f"Sentiment: {response.sentiment}")
print(f"Pros: {response.pros}")  
print(f"Cons: {response.cons}")
