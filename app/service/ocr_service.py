import io
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from PIL import Image

# Tesseract OCR imports
try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract not available. Install with: pip install pytesseract pillow")


class OcrService:
    """Tesseract OCR service for extracting text from image URLs"""

    def __init__(
        self,
        max_workers: int = 6,
        tesseract_path: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ):
        self.max_workers = max_workers
        self.languages = languages or ["eng", "vie"]
        self.tesseract_available = False
        self.language_string = "+".join(languages)

        if TESSERACT_AVAILABLE:
            self._initialize_tesseract(tesseract_path)
        else:
            raise RuntimeError("Tesseract OCR not available. Install pytesseract and pillow.")

        if self.tesseract_available:
            self.ocr_executor = ThreadPoolExecutor(max_workers=max_workers)

    def _initialize_tesseract(self, tesseract_path: Optional[str]):
        """Ensure Tesseract is installed and required languages are available"""
        try:
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

            available_languages = pytesseract.get_languages(config="")
            print(f"Available Tesseract languages: {available_languages}")

            missing_langs = [lang for lang in self.languages if lang not in available_languages]
            if missing_langs:
                raise RuntimeError(
                    f"Missing required Tesseract language packs: {missing_langs}\n"
                    f"Install them before running OCR.\n"
                    f"Ubuntu/Debian: sudo apt install tesseract-ocr-vie\n"
                    f"macOS (brew): brew install tesseract-lang"
                )

            self.tesseract_available = True
            print(f"Tesseract OCR initialized with languages: {self.language_string}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Tesseract: {e}")

    def _fetch_image_from_url(self, url: str) -> Optional[np.ndarray]:
        """Fetch image from URL and decode to OpenCV format"""
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"Failed to fetch image {url}: {e}")
            return None

    def _preprocess_image_for_ocr(self, url: str) -> Optional[np.ndarray]:
        """Preprocess image for better OCR accuracy"""
        try:
            image = self._fetch_image_from_url(url)
            if image is None:
                return None

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            thresh = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            return cleaned
        except Exception as e:
            print(f"Image preprocessing error for {url}: {e}")
            return None

    def _extract_text_from_url(self, url: str, r_id: int) -> str:
        """Extract text from an image URL using Tesseract OCR"""
        if not self.tesseract_available:
            return ""

        try:
            processed_image = self._preprocess_image_for_ocr(url)

            if processed_image is None:
                # fallback: use PIL directly
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                image = Image.open(io.BytesIO(resp.content))
            else:
                image = Image.fromarray(processed_image)

            custom_config = f"--oem 3 --psm 6 -l {self.language_string}"
            text = pytesseract.image_to_string(image, config=custom_config)
            return self._clean_extracted_text(text), r_id

        except Exception as e:
            print(f"Tesseract OCR error for {url}: {e}")
            return ""

    def _clean_extracted_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        cleaned = " ".join(raw_text.split())
        words = cleaned.split()
        filtered = [w for w in words if len(w) > 1 or w.isdigit()]
        return " ".join(filtered).strip()

    async def process_urls_batch(
        self,
        result_ids: List[int],
        urls: List[str]
    ) -> List[str]:
        """OCR for a batch of image URLs asynchronously"""
        if not self.tesseract_available:
            return [""] * len(urls)

        loop = asyncio.get_event_loop()
        ocr_tasks = [
            loop.run_in_executor(self.ocr_executor, self._extract_text_from_url, url, r_id)
            for url, r_id in zip(urls, result_ids)
        ]
        results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
        return [r if not isinstance(r, Exception) else "" for r in results]

    # def extract_text_sync(self, url: str) -> str:
    #     """Synchronous text extraction for single URL"""
    #     return self._extract_text_from_url(url)

    def is_available(self) -> bool:
        return self.tesseract_available

    def __del__(self):
        if hasattr(self, "ocr_executor"):
            self.ocr_executor.shutdown(wait=True)

