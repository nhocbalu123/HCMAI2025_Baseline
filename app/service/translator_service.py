import os
import google.genai as genai
from google.genai import types


class TranslatorService:
    def __init__(self, model_name=None):
        self._model_name = model_name or "gemini-2.0-flash-exp"
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and api_key != "dummy_key_for_dev":
            try:
                self._client = genai.Client(api_key=api_key)
                self._enabled = True
                print("Google AI translator service initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize Google AI client: {e}")
                self._client = None
                self._enabled = False
        else:
            if api_key == "dummy_key_for_dev":
                print("Using dummy Google API key. Translator service will return original queries.")
            else:
                print("Warning: GOOGLE_API_KEY not found. Translator service will be disabled.")
            self._client = None
            self._enabled = False
    
    def perform(self, query) -> str | None:
        if not self._enabled or not self._client:
            # Return the original query if translator is not available
            print("Translator service is not available, returning original query")
            return query
            
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
            print(f"An error occurred during translation: {e}")
            # Return the original query if translation fails
            return query
