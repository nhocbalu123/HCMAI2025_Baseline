import os
import google.genai as genai
from google.genai import types


class TranslatorService:
    def __init__(self, model_name=None):
        self._model_name = model_name or "gemini-2.0-flash-exp"
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    def perform(self, query) -> str | None:
        try:
            prompt = f"Translate the following Vietnamese text to English without additional explain:\n\n{query}"
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for consistent translation
                    max_output_tokens=2048,
                    top_p=0.8
                )
            )

            return response.text

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
