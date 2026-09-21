import os
import warnings
import logging
from dotenv import load_dotenv

# Silence SDK noise before importing
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from google import genai
from google.genai import types

# Primary model — falls back to lite on 503
GEMINI_MODEL = ["gemini-flash-lite-latest", "gemini-flash-latest"] 

class AiModel:
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY in .env file.")
        self.client = genai.Client(api_key=api_key)
        # 2. DEFAULT TO THE FIRST ITEM IN YOUR LIST
        self.model  = GEMINI_MODEL[0] 
        print(f"Gemini ready ({self.model}).")

    def _call(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.2) -> str:
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        last_err = None
        # 3. THIS WILL NOW PROPERLY LOOP THROUGH BOTH STRINGS
        for model in GEMINI_MODEL:
            try:
                resp = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config,
                )
                self.model = model   
                return resp.text.strip()
            except Exception as e:
                # 4. CATCH BOTH 429 (RATE LIMITS) AND 503 (UNAVAILABLE)
                if any(err in str(e) for err in ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"]):
                    last_err = e
                    continue   
                raise
        raise RuntimeError(f"All Gemini models unavailable: {last_err}")
    def rewrite_query(self, question: str) -> str:
        """
        Convert a conversational question into a formal legal search query.

        "can my landlord kick me out?"
        → "grounds for termination of lease contract by landlord"
        """
        prompt = (
            "Task: convert a user question into a formal Ethiopian legal search query.\n"
            "Rules:\n"
            "- Use precise legal terminology\n"
            "- Keep it under 12 words\n"
            "- Output the query ONLY, no explanation\n\n"
            f"User question: {question}\n"
            "Formal legal query:"
        )
        try:
            result = self._call(prompt, max_tokens=60, temperature=0.0)
            # Sanity check: must be shorter and different enough to be useful
            if result and result.lower() != question.lower() and len(result) > 5:
                return result
        except Exception:
            pass
        return question

    def ask_a_question(self, prompt: str) -> str:
        """Send a full RAG prompt and return the model's answer."""
        return self._call(prompt, max_tokens=1024, temperature=0.2)

    def full_prompt_for_rag(self, relevent_sections: str, question_prompt: str) -> str:
        """Build a grounded RAG prompt with citation instructions."""
        return (
            "You are an AI assistant specialising in Ethiopian law.\n"
            "Answer the question below using ONLY the provided legal text.\n"
            "For every claim cite the source as: (Article N, [Document Name]).\n"
            "If the answer is not in the text reply: "
            "'The document does not contain information on this topic.'\n\n"
            f"Legal Text:\n---\n{relevent_sections}\n---\n\n"
            f"Question: {question_prompt}\n\n"
            "Answer:"
        )
