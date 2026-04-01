"""
Vision Language Model (VLM) wrapper using Google Gemini 1.5 Flash.

Responsible for generating descriptive text summaries of images extracted
from EV technical documents (e.g., battery pack diagrams, performance charts,
thermal management schematics).
"""

import base64
import io
import logging
from typing import Optional

import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)

_VLM_SYSTEM_PROMPT = """You are a senior automotive engineer specialising in Electric Vehicles (EV).
You are analysing images extracted from TATA Motors EV technical documentation.

Your task: produce a concise but information-dense description of the image that will be used
as a text chunk in a retrieval-augmented generation (RAG) system.

Include:
- Type of visual (diagram, chart, table, photograph, schematic, etc.)
- All visible numerical values, labels, units, and legends
- Key technical insights or trends visible in the image
- Any component names, part numbers, or abbreviations

Be precise. A user querying the RAG system should be able to answer domain-specific questions
from your description alone. If the image is unclear or contains no useful technical content,
state that briefly.
"""


class VisionModel:
    """Wraps Gemini 1.5 Flash for multimodal (image → text) summarisation."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
        """
        Initialise the VLM.

        Args:
            api_key:    Google AI Studio API key.
            model_name: Gemini model identifier.
        """
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_VLM_SYSTEM_PROMPT,
        )
        self._model_name = model_name
        logger.info("VisionModel initialised with model '%s'.", model_name)

    def summarise_image(
        self,
        image_b64: str,
        image_ext: str = "png",
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a text summary of a base64-encoded image.

        Args:
            image_b64:  Base64-encoded image bytes.
            image_ext:  Image format extension (png, jpg, jpeg, etc.).
            context:    Optional surrounding text context from the same PDF page.

        Returns:
            A text description suitable for embedding and retrieval.
        """
        try:
            image_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if needed (e.g., CMYK PDFs)
            if pil_image.mode not in ("RGB", "RGBA"):
                pil_image = pil_image.convert("RGB")

            prompt_parts: list = []
            if context:
                prompt_parts.append(
                    f"The following image appears on a page with this surrounding text:\n\n"
                    f"---\n{context[:500]}\n---\n\n"
                    "Now describe the image:"
                )
            else:
                prompt_parts.append("Describe this technical image from an EV document:")

            prompt_parts.append(pil_image)

            response = self._model.generate_content(prompt_parts)
            summary = response.text.strip()
            logger.debug("VLM produced %d-char summary.", len(summary))
            return summary

        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM summarisation failed: %s", exc)
            return f"[Image summary unavailable: {exc}]"

    @property
    def model_name(self) -> str:
        """Return the underlying Gemini model identifier."""
        return self._model_name
