import os
import json
import litellm
from pydantic import BaseModel, ValidationError


class ItineraryRequest(BaseModel):
    destination: str


class ItineraryResponse(BaseModel):
    destination: str
    price_range: str
    ideal_visit_times: list[str]
    top_attractions: list[str]


def get_itinerary(destination: str) -> ItineraryResponse:
    AZUREAI_OPENAI_BASE_URL = os.environ.get("AZUREAI_OPENAI_BASE_URL")
    AZUREAI_OPENAI_API_KEY = os.environ.get("AZUREAI_OPENAI_API_KEY")
    model = os.environ.get("AZUREAI_OPENAI_MODEL", "azure/gpt-4.1-mini")

    if not AZUREAI_OPENAI_BASE_URL:
        raise RuntimeError("Missing AZUREAI_OPENAI_BASE_URL env var")
    if not AZUREAI_OPENAI_API_KEY:
        raise RuntimeError("Missing AZUREAI_OPENAI_API_KEY env var")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a travel itinerary generator.\n"
                "Respond ONLY with valid JSON matching this schema:\n"
                "{\n"
                '  "destination": string,\n'
                '  "price_range": string,\n'
                '  "ideal_visit_times": string[],\n'
                '  "top_attractions": string[]\n'
                "}\n"
                "Do not include explanations, markdown, or extra text."
            ),
        },
        {
            "role": "user",
            "content": f"Generate a travel itinerary for {destination}.",
        },
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=AZUREAI_OPENAI_API_KEY,
        api_base=AZUREAI_OPENAI_BASE_URL,
        response_format={"type": "json_object"},
    )
    print("Full model response:", response, flush=True)

    raw_content = response.choices[0].message.content
    print("Raw model response:", raw_content, flush=True)
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{raw_content}") from e

    try:
        return ItineraryResponse.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"JSON did not match schema:\n{data}") from e