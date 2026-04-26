# Structured Output
In Langchain, Structured refers to the prctice of having language models return responses in a well-degined data format (for example, JSON) rather than free form text. This makes model output easier to parse and work with programmatically.

### [Prompt] 
can you create a one-day travel itinerary for Paris?

### [LLM's Unstructured Response]
Here's a suggested itinerary: Morning: Visit the Effel Tower.
Afternoon: Walk through the Louvre Museum.
Evening: Enjoy dinner at a Seine riverside cafe.

### [JSON enforced output]
```
[
    {"time": "Morning", "activity": "Visit the Effel Tower."},
    {"time": "AfterNoon", "activity": "Walk through the Louvre Museum."},
    {"time": "Evening", "activity": "Enjoy dinner at a Seine riverside cafe."}
]
```
