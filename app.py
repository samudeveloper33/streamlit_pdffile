import streamlit as st
import os
import re
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Tuple
import tempfile
import logging
from datetime import datetime
import warnings

# Suppress all PaddleOCR and Paddle warnings before imports
warnings.filterwarnings("ignore", message=".*ccache.*")
warnings.filterwarnings("ignore", message=".*mean pooling instead of CLS embedding.*")
os.environ['GLOG_minloglevel'] = '2'  # Suppress Google logging (0=INFO, 1=WARNING, 2=ERROR)
os.environ['FLAGS_print_model_net_proto'] = '0'  # Suppress Paddle model printing

# PDF and Image Processing
from pdf2image import convert_from_path
from PIL import Image
import cv2
from paddleocr import PaddleOCR

# Text Processing

# Translation
try:
    from google.cloud import translate_v2
    HAS_GOOGLE_TRANSLATE = True
except (ImportError, Exception):
    # Suppress all errors (including credentials file not found)
    HAS_GOOGLE_TRANSLATE = False

# Embeddings and Vector Store
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import google.generativeai as genai

# Environment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for terminal output (MUST BE BEFORE API CONFIG)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configure API keys
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Global constants
COLLECTION_NAME = ""

# Remove quotes if they exist in the .env file
if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip('"').strip("'")
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info(f"✅ Gemini API configured (Key: {GEMINI_API_KEY[:10]}...)")
else:
    st.error("GEMINI_API_KEY not found in .env file")
    logger.error("❌ GEMINI_API_KEY not found in .env file")

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,  # Recommended for performance
        timeout=30.0  # Increase timeout from default 5 seconds to 30 seconds
    )
    logger.info("✅ Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Qdrant client: {str(e)}")
    st.error(f"Qdrant connection error: {str(e)}")
    qdrant_client = None


# Configure Streamlit page
st.set_page_config(
    page_title="PaddleOCR + RAG Processing",
    page_icon="📄",
    layout="wide"
)

# Initialize PaddleOCR with multilingual support
def get_paddle_ocr():
    """
    Initialize PaddleOCR with Tamil and English support.
    """
    logger.info("🤖 Initializing PaddleOCR (Tamil + English)...")
    try:
        ocr = PaddleOCR(
            use_textline_orientation=True, 
            lang='ta'
        )
        logger.info("✅ PaddleOCR initialized successfully.")
        return ocr
    except Exception as e:
        logger.error(f"❌ Failed to initialize PaddleOCR: {str(e)}")
        st.error(f"Fatal Error: Could not initialize the OCR engine. Please check logs. Error: {e}")
        return None

# Initialize the embedding model
@st.cache_resource
def get_embedding_model():
    """
    Initialize the embedding model with fallback options.
    """
    logger.info("🧠 Initializing embedding model...")
    try:
        # Use FastEmbed for faster inference with correct model name
        model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
        logger.info("✅ FastEmbed model (intfloat/multilingual-e5-large) loaded successfully")
        return model
    except Exception as e:
        logger.error(f"❌ FastEmbed model loading failed: {str(e)}")
        logger.info("⚠️ Attempting to load alternative multilingual model...")
        try:
            # Fallback to paraphrase-multilingual-mpnet-base-v2
            model = TextEmbedding(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
            logger.info("✅ Alternative model loaded: paraphrase-multilingual-mpnet-base-v2")
            return model
        except Exception as fallback_error:
            logger.error(f"❌ Alternative model also failed: {str(fallback_error)}")
            logger.info("⚠️ Using default FastEmbed model...")
            model = TextEmbedding()
            logger.info("✅ Default FastEmbed model loaded")
            return model

# Initialize models
paddle_ocr = get_paddle_ocr()
embedding_model = get_embedding_model()

def detect_language(text: str) -> str:
    """Detect if text is Tamil or English based on character ranges."""
    tamil_count = 0
    english_count = 0
    
    for char in text:
        # Tamil Unicode range: U+0B80 to U+0BFF
        if '\u0B80' <= char <= '\u0BFF':
            tamil_count += 1
        # English letters
        elif char.isalpha():
            english_count += 1
    
    # If more than 30% Tamil characters, consider it Tamil
    if tamil_count > 0 and tamil_count > english_count:
        return "ta"
    return "en"

def translate_text(text: str, source_lang: str = "ta", target_lang: str = "en") -> str:
    """
    Translate text using Google Translate API (with fallback to Gemini).
    
    Args:
        text: Text to translate
        source_lang: Source language code (default: "ta" for Tamil)
        target_lang: Target language code (default: "en" for English)
    
    Returns:
        Translated text, or original text if translation fails
    """
    if not text or not text.strip():
        return text
    
    try:
        # Try using Google Cloud Translation API first
        if HAS_GOOGLE_TRANSLATE:
            try:
                translate_client = translate_v2.Client()
                result = translate_client.translate_text(
                    text,
                    source_language=source_lang,
                    target_language=target_lang
                )
                return result.get('translatedText', text)
            except Exception as e:
                # Silently skip if Google Cloud API is not available
                logger.debug(f"⚠️ Google Cloud Translation API unavailable: {str(e)}")
        
        # Fallback: Use Gemini for translation
        # Get available models and try each one
        available_models = list_available_gemini_models()
        
        if not available_models:
            # Use default model list for translation
            available_models = [
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
        
        prompt = f"""Translate the following text from {source_lang.upper()} to {target_lang.upper()}. 
Only provide the translated text without any explanation or additional content.

Text to translate:
{text}"""
        
        last_error = None
        for model_name in available_models:
            try:
                logger.info(f"🤖 Attempting translation with model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                translated = response.text.strip()
                logger.info(f"✅ Translation successful using {model_name}")
                return translated
            except Exception as model_error:
                last_error = model_error
                logger.debug(f"⚠️ Model {model_name} failed for translation: {str(model_error)}")
                continue
        
        # If all models failed
        logger.error(f"❌ Translation failed with all available models. Last error: {str(last_error)}")
        return text
        
    except Exception as e:
        logger.warning(f"⚠️ Translation error: {str(e)}")
        return text


def get_localized_response(response_text: str, target_language: str = "en") -> str:
    """
    Translate response to match the user's query language.
    
    Args:
        response_text: The response text (in English)
        target_language: Target language ("ta" for Tamil, "en" for English)
    
    Returns:
        Response in the target language
    """
    if target_language == "en":
        # Response is already in English
        return response_text
    
    if target_language == "ta":
        # Translate English response to Tamil
        try:
            tamil_response = translate_text(response_text, source_lang="en", target_lang="ta")
            return tamil_response
        except Exception as e:
            logger.warning(f"⚠️ Failed to translate response to Tamil: {str(e)}")
            return response_text
    
    return response_text




def remove_watermarks(text: str) -> str:
    """Remove common watermark patterns from text."""
    # Common watermark patterns
    watermark_patterns = [
        r'confidential',
        r'draft',
        r'do not distribute',
        r'copyright.*\d{4}',
        r'all rights reserved',
        r'watermark',
    ]
    
    cleaned_text = text
    for pattern in watermark_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text


def clean_ocr_text(text: str) -> str:
    """Clean and correct OCR text with comprehensive Tamil error corrections."""
    if not text or text.strip() == "":
        return ""
    
    # Remove watermarks
    text = remove_watermarks(text)
    
    # Normalize Unicode to NFD (decomposed) and then back to NFC (composed)
    # This fixes issues with Tamil vowel signs and combining characters
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width characters that can corrupt the text
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\u200c', '')  # Zero-width non-joiner
    text = text.replace('\u200d', '')  # Zero-width joiner
    text = text.replace('\ufeff', '')  # Zero-width no-break space (BOM)
    
    lines = text.split('\n')
    cleaned_lines = []
    
    # Process each line
    for line in lines:
        # Remove extra spaces but preserve the line
        cleaned_line = ' '.join(line.split())
        # Keep the line even if empty to preserve document structure
        cleaned_lines.append(cleaned_line)

    # Rejoin lines while preserving structure
    final_text = '\n'.join(cleaned_lines)
    
    # Remove consecutive empty lines (more than 2)
    final_text = re.sub(r'\n\n\n+', '\n\n', final_text)
    
    logger.info(f"✅ Text cleaned - Original: {len(text)} chars, Cleaned: {len(final_text)} chars")
    
    return final_text.strip()




def gemini_enhance_tamil_ocr(raw_ocr_text: str, use_enhanced_prompt: bool = True) -> str:
    """
    Use Gemini to enhance and correct Tamil OCR text.
    
    Args:
        raw_ocr_text: The raw OCR output text
        use_enhanced_prompt: If True, uses the enhanced accuracy prompt; otherwise uses the perfect prompt
    
    Returns:
        Enhanced and corrected Tamil text
    """
    if not raw_ocr_text or raw_ocr_text.strip() == "":
        return raw_ocr_text
    
    try:
        logger.info("🤖 Using Gemini to enhance Tamil OCR accuracy...")
        
        # Prompt for natural Tamil document layout matching
        prompt = f"""Fix this Tamil OCR text to match how it should appear in the original document.

INSTRUCTIONS:
1. **Match Image Content**: The text must match the original image content exactly.
2. **Fix Word Order**: Ensure dates appear before days and time words appear before numbers.
3. **Preserve Reference Numbers**: Match file numbers and codes (e.g., headers with dots) EXACTLY character-by-character. Do NOT auto-correct them to common prefixes.
4. **Fix Abbreviation Typos**: Correct OCR errors in other abbreviations but DO NOT expand them.
5. **Remove Artifacts**: Remove page numbers, arrows, and garbled nonsense.
6. **Line Breaks**: Keep original line breaks, but join words/phrases if they are incorrectly split across lines.
7. **No Omissions**: Include ALL valid text from the image.

Raw OCR Text:
{raw_ocr_text}

Corrected Text:"""
        
        # Try available Gemini models
        available_models = list_available_gemini_models()
        
        if not available_models:
            # Use default model list
            available_models = [
                'gemini-1.5-flash-latest',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro'
            ]
        
        last_error = None
        for model_name in available_models:
            try:
                logger.info(f"🤖 Attempting OCR enhancement with model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                enhanced_text = response.text.strip()
                logger.info(f"✅ OCR enhancement successful using {model_name}")
                logger.info(f"📊 Enhanced text length: {len(enhanced_text)} characters")
                return enhanced_text
            except Exception as model_error:
                last_error = model_error
                logger.debug(f"⚠️ Model {model_name} failed for OCR enhancement: {str(model_error)}")
                continue
        
        # If all models failed, return original text
        logger.warning(f"⚠️ Gemini OCR enhancement failed with all models. Last error: {str(last_error)}")
        logger.warning(f"⚠️ Returning original OCR text without Gemini enhancement")
        return raw_ocr_text
        
    except Exception as e:
        logger.warning(f"⚠️ Gemini OCR enhancement error: {str(e)}")
        logger.warning(f"⚠️ Returning original OCR text")
        return raw_ocr_text



def gentle_preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Gentle image preprocessing that preserves color information for PaddleOCR.
    Avoids aggressive binary conversion that destroys text information.
    """
    try:
        logger.info("🖼️ Gentle preprocessing for OCR (color-preserving)...")
        
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        logger.info(f"📐 Original image size: {img_array.shape[1]}x{img_array.shape[0]}")
        
        # Gentle denoising - preserve details
        logger.info("🧹 Gentle denoising...")
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 8, 8, 7, 21)
        
        # Light contrast enhancement using CLAHE
        logger.info("💡 Gentle contrast enhancement...")
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        logger.info("✅ Gentle preprocessing complete")
        return Image.fromarray(enhanced)
        
    except Exception as e:
        logger.warning(f"⚠️ Gentle preprocessing failed: {str(e)}, using original image")
        return image


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Ultra-optimized image preprocessing for Tamil OCR.
    Handles poor quality scans with advanced techniques.
    """
    try:
        logger.info("🖼️ Starting advanced Tamil OCR preprocessing...")
        
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if not already
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        logger.info(f"📐 Original image size: {gray.shape[1]}x{gray.shape[0]}")
        
        # Step 1: Extreme upscaling for poor quality images
        height, width = gray.shape
        target_height = 3000  # Very high resolution for Tamil complexity
        if height < target_height:
            scale_factor = max(target_height // height, 3)
            logger.info(f"🔍 Extreme upscaling {scale_factor}x for poor quality Tamil text...")
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
            logger.info(f"✅ Upscaled to {gray.shape[1]}x{gray.shape[0]}")
        
        # Step 2: Bilateral filter for strong edge preservation
        logger.info("🧹 Bilateral filtering (edge-preserving noise reduction)...")
        bilateral = cv2.bilateralFilter(gray, 13, 100, 100)
        
        # Step 3: Multiple rounds of denoising for very poor text
        logger.info("🧹 Heavy denoising (multiple passes)...")
        denoised = cv2.fastNlMeansDenoising(bilateral, None, h=15, templateWindowSize=7, searchWindowSize=21)
        denoised = cv2.fastNlMeansDenoising(denoised, None, h=12, templateWindowSize=7, searchWindowSize=21)
        
        # Step 4: Gamma correction to fix dark/light issues
        logger.info("💡 Gamma correction for brightness normalization...")
        gamma = 1.3  # Adjust brightness
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, table)
        
        # Step 5: Histogram equalization
        logger.info("📊 CLAHE enhancement...")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12, 12))
        enhanced = clahe.apply(gamma_corrected)
        
        # Step 6: Remove very small noise with opening
        logger.info("🔧 Morphological opening (noise removal)...")
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        opened = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Step 7: Close small holes in characters
        logger.info("🔧 Morphological closing (fill holes in characters)...")
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Step 8: Aggressive adaptive thresholding for Tamil
        logger.info("🎯 Aggressive adaptive thresholding for Tamil text...")
        binary = cv2.adaptiveThreshold(
            closed, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            21,  # Large block size for Tamil complexity
            4    # Higher constant for better separation
        )
        
        # Step 9: Ensure dark text on light background
        black_pixels = np.sum(binary == 0)
        white_pixels = np.sum(binary == 255)
        
        if black_pixels > white_pixels:
            logger.info("🔄 Inverting to dark text on light background...")
            binary = cv2.bitwise_not(binary)
        
        # Step 10: Slight dilation to connect broken characters
        logger.info("🔗 Dilation to reconnect characters...")
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.dilate(binary, kernel_dilate, iterations=1)
        
        # Step 11: Final cleanup
        logger.info("🧹 Final cleanup...")
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean, iterations=1)
        
        processed_image = Image.fromarray(binary)
        
        logger.info("✅ Ultra-advanced preprocessing complete")
        return processed_image
        
    except Exception as e:
        logger.warning(f"⚠️ Image preprocessing failed: {str(e)}, using original image")
        return image


def pdf_page_to_image(pdf_path: str, page_num: int, dpi: int = 300) -> Image.Image:
    """Convert a specific PDF page to high-quality image."""
    try:
        logger.info(f"📄 Converting Page {page_num} to image (DPI: {dpi})...")
        images = convert_from_path(
            pdf_path,
            first_page=page_num,
            last_page=page_num,
            dpi=dpi,
            fmt='png'
        )
        if images:
            logger.info(f"✅ Page {page_num} converted to image successfully")
            return images[0]
        else:
            logger.error(f"❌ Failed to convert page {page_num} - No image generated")
            return None
    except Exception as e:
        logger.error(f"❌ Error converting page {page_num}: {str(e)}")
        st.error(f"Error converting page {page_num}: {str(e)}")
        return None


def perform_ocr(image: Image.Image, preprocess: bool = False) -> str:
    """
    Perform OCR using PaddleOCR on an image with optimized preprocessing for Tamil.
    
    NOTE: This function ONLY uses PaddleOCR. No other OCR engines are used.
    The ocr() method returns bounding boxes which are used to sort text spatially
    (top-to-bottom, left-to-right) ensuring correct Tamil word order.
    
    Parameters:
        image: PIL Image to process
        preprocess: Whether to apply gentle image preprocessing (disabled by default - PaddleOCR works well on original images)
    
    Returns:
        Extracted text from the image in correct spatial order
    """
    try:
        logger.info(f"🔍 Performing OCR using PaddleOCR...")
        
        # Convert PIL Image to numpy array first
        img_array = np.array(image)
        
        # Ensure the image is in the correct format (RGB)
        if len(img_array.shape) == 2:
            # Grayscale - convert to RGB for PaddleOCR
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            # RGBA - convert to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            # Single channel - convert to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Resize image if it exceeds PaddleOCR's max_side_limit (4000 pixels)
        height, width = img_array.shape[:2]
        max_side_limit = 4000
        
        if width > max_side_limit or height > max_side_limit:
            # Calculate scaling factor to fit within limit while maintaining aspect ratio
            scale_factor = max_side_limit / max(width, height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            logger.info(f"📐 Resizing image from {width}x{height} to {new_width}x{new_height} to fit PaddleOCR limits...")
            img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Perform OCR with PaddleOCR
        # PaddleOCR: The ONLY OCR engine used in this application
        logger.info("📊 PaddleOCR is processing the image...")
        logger.info(f"📊 Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
        logger.info(f"📊 Image value range: min={img_array.min()}, max={img_array.max()}")
        
        # Use ocr() method to get bounding boxes (predict() doesn't return them)
        # Suppress deprecation warning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Please use.*predict.*")
            results = paddle_ocr.ocr(img_array)
        
        logger.info(f"📊 OCR Results type: {type(results)}, Length: {len(results) if results else 0}")
        if results and len(results) > 0:
            logger.info(f"📊 First result structure: {type(results[0])}, has content: {bool(results[0])}")
        
        # Check if results are empty and try with preprocessing if needed
        if not results or len(results) == 0 or (len(results) > 0 and not results[0]):
            logger.warning("⚠️ PaddleOCR returned empty results on first attempt")
            logger.info("🔧 Retrying with image preprocessing...")
            
            # Apply gentle preprocessing for better text detection
            preprocessed_img = gentle_preprocess_for_ocr(image)
            preprocessed_array = np.array(preprocessed_img)
            
            # Ensure correct format
            if len(preprocessed_array.shape) == 2:
                preprocessed_array = cv2.cvtColor(preprocessed_array, cv2.COLOR_GRAY2RGB)
            elif len(preprocessed_array.shape) == 3 and preprocessed_array.shape[2] == 4:
                preprocessed_array = cv2.cvtColor(preprocessed_array, cv2.COLOR_RGBA2RGB)
            
            # Retry OCR
            logger.info("📊 Retrying PaddleOCR with preprocessed image...")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Please use.*predict.*")
                results = paddle_ocr.ocr(preprocessed_array)
            
            logger.info(f"📊 Retry - OCR Results type: {type(results)}, Length: {len(results) if results else 0}")
        
        # Extract text from results with spatial sorting
        text = ""
        text_with_coords = []
        
        # Handle ocr() format which returns [[bbox, (text, confidence)], ...]
        if results and len(results) > 0:
            page_result = results[0]  # First page
            
            if page_result:
                # Handle dictionary format (PaddleOCR v2.7+ sometimes returns dict)
                if isinstance(page_result, dict):
                    logger.info(f"📊 PaddleOCR returned dictionary. Keys: {list(page_result.keys())}")
                    
                    # Log types/lengths of values for debugging
                    for k, v in page_result.items():
                        v_type = type(v)
                        v_len = len(v) if hasattr(v, '__len__') else 'N/A'
                        logger.info(f"   Key: {k}, Type: {v_type}, Len: {v_len}")

                    # Try to find boxes
                    dt_boxes = page_result.get('dt_boxes')
                    if dt_boxes is None or len(dt_boxes) == 0:
                        dt_boxes = page_result.get('rec_boxes')
                    if dt_boxes is None or len(dt_boxes) == 0:
                        dt_boxes = page_result.get('boxes')
                    if dt_boxes is None or len(dt_boxes) == 0:
                        dt_boxes = page_result.get('rec_polys')  # New format uses rec_polys
                    
                    # Try to find text results - NEW FORMAT PRIORITY
                    rec_res = page_result.get('rec_texts')  # NEW: Try this first for new PaddleOCR versions
                    if rec_res is None or len(rec_res) == 0:
                        rec_res = page_result.get('rec_res')
                    if rec_res is None or len(rec_res) == 0:
                        rec_res = page_result.get('rec_text')
                    if rec_res is None or len(rec_res) == 0:
                        rec_res = page_result.get('text')

                    # Ensure they are lists
                    dt_boxes = dt_boxes if dt_boxes is not None else []
                    rec_res = rec_res if rec_res is not None else []

                    # Convert numpy arrays to lists
                    if isinstance(dt_boxes, np.ndarray):
                        dt_boxes = dt_boxes.tolist()
                    if isinstance(rec_res, np.ndarray):
                        rec_res = rec_res.tolist()

                    logger.info(f"   Final extraction - Boxes: {len(dt_boxes)}, Text: {len(rec_res)}")

                    if len(rec_res) > 0:
                        logger.info(f"   First text item: {rec_res[0]}")

                    # Reconstruct into list of [bbox, (text, conf)]
                    # Handle different text item formats
                    reconstructed_items = []
                    for i, box in enumerate(dt_boxes):
                        if i < len(rec_res):
                            text_item = rec_res[i]
                            # Extract text based on format
                            if isinstance(text_item, str):
                                # Simple string format
                                detected_text = text_item
                                confidence = 0.99
                            elif isinstance(text_item, (list, tuple)) and len(text_item) >= 1:
                                # Tuple format (text, confidence)
                                detected_text = text_item[0]
                                confidence = text_item[1] if len(text_item) > 1 else 0.99
                            else:
                                detected_text = str(text_item)
                                confidence = 0.99
                            
                            if detected_text and str(detected_text).strip():
                                reconstructed_items.append([box, (detected_text, confidence)])
                    
                    page_result = reconstructed_items
                    
                    # Check for mismatch with no text (0 text but has boxes)
                    if len(page_result) == 0 and len(dt_boxes) > 0:
                        logger.error(f"⚠️ Critical mismatch: {len(dt_boxes)} boxes detected but NO valid text extracted")
                        logger.error(f"   🔍 dt_boxes keys: {list(page_result.keys()) if isinstance(page_result, dict) else 'N/A'}")
                        logger.error(f"   🔍 Checking original page_result for text...")
                        
                        logger.info("🔧 Triggering retry with preprocessing due to text extraction failure...")
                        
                        # Apply gentle preprocessing for better text detection
                        preprocessed_img = gentle_preprocess_for_ocr(image)
                        preprocessed_array = np.array(preprocessed_img)
                        
                        # Ensure correct format
                        if len(preprocessed_array.shape) == 2:
                            preprocessed_array = cv2.cvtColor(preprocessed_array, cv2.COLOR_GRAY2RGB)
                        elif len(preprocessed_array.shape) == 3 and preprocessed_array.shape[2] == 4:
                            preprocessed_array = cv2.cvtColor(preprocessed_array, cv2.COLOR_RGBA2RGB)
                        
                        # Retry OCR
                        logger.info("📊 Retrying PaddleOCR with preprocessed image...")
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message=".*Please use.*predict.*")
                            retry_results = paddle_ocr.ocr(preprocessed_array)
                        
                        logger.info(f"📊 Retry - OCR Results type: {type(retry_results)}, Length: {len(retry_results) if retry_results else 0}")
                        
                        # Check retry results
                        retry_has_text = False
                        if retry_results and len(retry_results) > 0 and retry_results[0]:
                            retry_page = retry_results[0]
                            if isinstance(retry_page, list) and len(retry_page) > 0:
                                retry_has_text = True
                            elif isinstance(retry_page, dict):
                                retry_rec_res = retry_page.get('rec_res') or retry_page.get('rec_text') or retry_page.get('text')
                                retry_has_text = retry_rec_res is not None and len(retry_rec_res) > 0
                        
                        # Use retry results if they have text
                        if retry_has_text:
                            results = retry_results
                            page_result = results[0]
                            logger.info("✅ Retry succeeded, using preprocessed results")
                        else:
                            logger.warning("❌ Retry also failed - trying aggressive preprocessing...")
                            
                            # Try even more aggressive preprocessing as last resort
                            try:
                                aggressive_img = preprocess_image_for_ocr(image)
                                aggressive_array = np.array(aggressive_img)
                                
                                if len(aggressive_array.shape) == 2:
                                    aggressive_array = cv2.cvtColor(aggressive_array, cv2.COLOR_GRAY2RGB)
                                elif len(aggressive_array.shape) == 3 and aggressive_array.shape[2] == 4:
                                    aggressive_array = cv2.cvtColor(aggressive_array, cv2.COLOR_RGBA2RGB)
                                
                                logger.info("📊 Retrying PaddleOCR with aggressive preprocessing...")
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", message=".*Please use.*predict.*")
                                    aggressive_results = paddle_ocr.ocr(aggressive_array)
                                
                                if aggressive_results and len(aggressive_results) > 0 and aggressive_results[0]:
                                    results = aggressive_results
                                    page_result = results[0]
                                    logger.info("✅ Aggressive preprocessing retry succeeded")
                                else:
                                    logger.warning("❌ Aggressive preprocessing also failed - possible model issue")
                                    # Last resort: try with language list instead of single language
                                    logger.warning("🔧 Attempting OCR with explicit language list...")
                                    try:
                                        # This might help if the single language model failed
                                        last_resort_results = paddle_ocr.ocr(img_array)
                                        if last_resort_results and len(last_resort_results) > 0 and last_resort_results[0]:
                                            results = last_resort_results
                                            page_result = results[0]
                                            logger.info("✅ Last resort OCR attempt succeeded")
                                    except Exception as lr_error:
                                        logger.error(f"❌ All OCR attempts failed: {str(lr_error)}")
                            except Exception as aggressive_error:
                                logger.warning(f"⚠️ Aggressive preprocessing error: {str(aggressive_error)}")

                elif isinstance(page_result, list):
                    # Already in list format [[bbox, (text, conf)], ...]
                    logger.info(f"📊 PaddleOCR returned list format with {len(page_result)} lines")
                else:
                    # Unknown format
                    logger.warning(f"⚠️ Unknown page_result format: {type(page_result)}")
                    page_result = []

                logger.info(f"📊 Found {len(page_result)} text lines using ocr() method")
                
                # Create list of (text, y_coord, x_coord) for sorting
                text_with_coords = []
                
                for idx, line in enumerate(page_result):
                    try:
                        if line and len(line) >= 2:
                            bbox = line[0]  # Bounding box coordinates
                            text_info = line[1]  # (text, confidence) tuple
                            
                            # Extract text
                            if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                                detected_text = text_info[0]
                            else:
                                detected_text = text_info
                            
                            if detected_text and str(detected_text).strip():
                                # Get top-left corner coordinates for sorting
                                # bbox can be in multiple formats
                                y_coord = idx  # Default to index if we can't extract
                                x_coord = 0
                                
                                try:
                                    if isinstance(bbox, (int, float)):
                                        # bbox is just an integer ID or coordinate - use index
                                        logger.debug(f"   📝 Line {idx}: bbox is numeric ({bbox}), using index for sorting")
                                        y_coord = idx
                                        x_coord = 0
                                    elif isinstance(bbox, np.ndarray):
                                        # numpy array format
                                        if bbox.ndim == 2 and bbox.shape[0] > 0:
                                            y_coord = float(bbox[0, 1])  # Row 0, Column 1
                                            x_coord = float(bbox[0, 0])  # Row 0, Column 0
                                        else:
                                            y_coord = idx
                                            x_coord = 0
                                    elif isinstance(bbox, (list, tuple)) and len(bbox) > 0:
                                        # List/tuple format - could be [[x1,y1], ...] or [x1, y1, ...]
                                        first_elem = bbox[0]
                                        if isinstance(first_elem, (list, tuple)) and len(first_elem) >= 2:
                                            # Format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                                            y_coord = float(first_elem[1])
                                            x_coord = float(first_elem[0])
                                        elif isinstance(first_elem, (int, float)):
                                            # Format: [x1, y1, x2, y2] or similar
                                            if len(bbox) >= 2:
                                                x_coord = float(bbox[0])
                                                y_coord = float(bbox[1])
                                            else:
                                                y_coord = idx
                                                x_coord = 0
                                        else:
                                            y_coord = idx
                                            x_coord = 0
                                    else:
                                        # Unknown format
                                        logger.debug(f"   📝 Line {idx}: unknown bbox format {type(bbox)}")
                                        y_coord = idx
                                        x_coord = 0
                                except (TypeError, IndexError, ValueError) as coord_error:
                                    logger.debug(f"   📝 Line {idx}: could not extract bbox coords ({coord_error}), using index")
                                    y_coord = idx
                                    x_coord = 0
                                
                                text_with_coords.append((detected_text, y_coord, x_coord))
                    except (IndexError, TypeError, AttributeError, KeyError) as e:
                        logger.warning(f"⚠️ Could not process line {idx}: {e}")
                        continue
        
        # Process extracted text
        if text_with_coords:
            # Sort by Y (top-to-bottom) first, then X (left-to-right)
            # Use a threshold for Y to group lines (15 pixels tolerance for same line)
            logger.info("🔄 Sorting text by spatial position for correct Tamil word order...")
            
            # Sort with better line grouping: first by Y coordinate (rounded to 20px), then by X coordinate
            # This keeps words on same line together
            text_with_coords.sort(key=lambda item: (round(item[1] / 20), item[2]))
            
            logger.info(f"✅ Sorted {len(text_with_coords)} text elements by position")
            
            # Extract sorted text and group by lines with better threshold
            extracted_lines = []
            current_line = ""
            current_y = -1
            y_threshold = 20  # Increased from 15 to 20 pixels for better line grouping
            
            for txt, y, x in text_with_coords:
                txt = str(txt).strip()
                if not txt:  # Skip empty texts
                    continue
                    
                # Check if we're still on the same line
                if current_y >= 0 and abs(y - current_y) > y_threshold:
                    # New line detected - save current line and start new one
                    if current_line.strip():
                        extracted_lines.append(current_line.strip())
                    current_line = txt
                    current_y = y
                else:
                    # Same line - append with space if needed
                    if current_line and not current_line.endswith(' '):
                        current_line += " "
                    current_line += txt
                    if current_y < 0:
                        current_y = y
            
            # Add last line
            if current_line.strip():
                extracted_lines.append(current_line.strip())
            
            # Join lines with proper newlines
            text = "\n".join(extracted_lines)
        else:
            logger.warning("⚠️ No text with coordinates found")
        
        if not text.strip():
            logger.warning(f"⚠️ No text was detected by PaddleOCR. OCR Resultss format: {type(results)}")

        
        logger.info(f"✅ PaddleOCR completed - Extracted {len(text)} characters")
        
        return text.strip()
    except Exception as e:
        logger.error(f"❌ PaddleOCR failed: {str(e)}")
        raise Exception(f"PaddleOCR failed: {str(e)}")


def generate_embedding(text: str, task_type: str = "retrieval_document") -> List[float]:
    """Generate embedding for text using FastEmbed multilingual model."""
    try:
        logger.info(f"🧠 Generating embedding (text length: {len(text)} chars)...")
        
        # Add task-specific prefix for E5 models (improves performance)
        if task_type == "retrieval_document":
            prefixed_text = f"passage: {text}"
        elif task_type == "retrieval_query":
            prefixed_text = f"query: {text}"
        else:
            prefixed_text = text
        
        # Generate embedding using FastEmbed
        # FastEmbed returns a generator, so we need to get the first result
        embeddings = list(embedding_model.embed([prefixed_text]))
        
        if not embeddings or len(embeddings) == 0:
            logger.error("❌ No embeddings generated")
            return None
            
        # Get the first embedding and convert to list
        embedding_array = embeddings[0]
        embedding_list = embedding_array.tolist() if hasattr(embedding_array, 'tolist') else list(embedding_array)
        
        logger.info(f"✅ Embedding generated (dimension: {len(embedding_list)})")
        return embedding_list
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {str(e)}")
        st.error(f"Error generating embedding: {str(e)}")
        return None


def create_qdrant_collection(dimension: int = 1024):
    """Create Qdrant collection if it doesn't exist.
    
    Note: FastEmbed intfloat/multilingual-e5-large produces 1024-dimensional embeddings.
    """
    if qdrant_client is None:
        logger.error("❌ Qdrant client not initialized")
        st.error("Qdrant client not available")
        return False
    
    try:
        logger.info(f"📦 Attempting to create Qdrant collection '{COLLECTION_NAME}'...")
        
        # Try to get existing collections
        try:
            collections = qdrant_client.get_collections().collections
            collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        except Exception as get_error:
            logger.warning(f"⚠️ Could not retrieve collections: {str(get_error)}")
            logger.info("ℹ️ Assuming collection doesn't exist, will attempt to create...")
            collection_exists = False
        
        if collection_exists:
            # Check if dimension matches
            try:
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                existing_dim = collection_info.config.params.vectors.size
                
                if existing_dim != dimension:
                    logger.warning(f"⚠️ Dimension mismatch! Existing: {existing_dim}, Required: {dimension}")
                    logger.info(f"🗑️ Deleting old collection '{COLLECTION_NAME}'...")
                    qdrant_client.delete_collection(COLLECTION_NAME)
                    st.warning(f"Deleted old collection (dimension mismatch: {existing_dim} → {dimension})")
                    collection_exists = False
            except Exception as info_error:
                logger.warning(f"⚠️ Could not get collection info: {str(info_error)}")
        
        if not collection_exists:
            logger.info(f"📦 Creating Qdrant collection '{COLLECTION_NAME}' with dimension {dimension}...")
            try:
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
                )
                logger.info(f"✅ Collection '{COLLECTION_NAME}' created successfully")
                st.success(f"Created collection: {COLLECTION_NAME} (dimension: {dimension})")
            except Exception as create_error:
                logger.error(f"❌ Failed to create collection: {str(create_error)}")
                st.error(f"Failed to create Qdrant collection: {str(create_error)}")
                return False
        else:
            logger.info(f"✅ Collection '{COLLECTION_NAME}' already exists with correct dimension")
        return True
    except Exception as e:
        logger.error(f"❌ Error in create_qdrant_collection: {str(e)}")
        st.error(f"Collection error: {str(e)}")
        return False


def store_in_qdrant(pdf_name: str, page_number: int, clean_text: str, embedding: List[float]) -> bool:
    """Store page data in Qdrant in a separate collection for each PDF."""
    if qdrant_client is None:
        logger.error(f"❌ Qdrant client not available for storing page {page_number}")
        return False
    
    try:
        # Create collection name from PDF name (sanitize for Qdrant)
        import re
        collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pdf_name.replace('.pdf', ''))
        collection_name = collection_name[:64]  # Qdrant collection name max length
        
        logger.info(f"💾 Storing page {page_number} from '{pdf_name}' in collection '{collection_name}'...")
        
        # Ensure the collection exists. `recreate_collection` is idempotent and will either
        # create the collection if it doesn't exist or do nothing if it does.
        # Use create_collection which does not delete an existing collection
        # This is critical for multi-page documents
        # Try to create the collection. If it already exists, we can ignore the error.
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1024,  # Dimension of the FastEmbed model
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Collection '{collection_name}' created.")
        except Exception as e:
            # If the collection already exists, an error will be raised.
            # We check for this specific case and ignore it.
            if "already exists" in str(e).lower():
                logger.info(f"Collection '{collection_name}' already exists. Continuing.")
            else:
                # For any other error, we re-raise it to be handled.
                raise e
        logger.info(f"✅ Collection '{collection_name}' is ready.")
        
        # Generate truly unique point ID using UUID based on page number
        import uuid
        unique_string = f"{pdf_name}_{page_number}_{hash(clean_text[:100]) if clean_text else 0}"
        point_id = int(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string).int % (10 ** 9))
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "page_number": page_number,
                "pdf_name": pdf_name,
                "clean_text": clean_text,
                "document_id": f"{pdf_name}_page_{page_number}"
            }
        )
        
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            logger.info(f"✅ Page {page_number} stored successfully in collection '{collection_name}' (ID: {point_id})")
            return True
        except Exception as upsert_error:
            logger.error(f"❌ Failed to upsert page {page_number}: {str(upsert_error)}")
            logger.warning(f"⚠️ Continuing without storing page {page_number} in Qdrant")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error storing page {page_number} in Qdrant: {str(e)}")
        st.error(f"Error storing in Qdrant: {str(e)}")
        return False


def check_page_exists(pdf_name: str, page_number: int) -> bool:
    """Check if a specific page from a PDF has already been processed and stored in Qdrant.
    
    Args:
        pdf_name: Name of the PDF file
        page_number: Page number to check
    
    Returns:
        True if the page exists in Qdrant, False otherwise
    """
    if qdrant_client is None:
        return False
    
    try:
        # Create collection name from PDF name (same logic as store_in_qdrant)
        import re
        collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pdf_name.replace('.pdf', ''))
        collection_name = collection_name[:64]  # Qdrant collection name max length
        
        # Check if collection exists
        try:
            collections = qdrant_client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                return False
        except Exception:
            return False
        
        # Search for the specific page using scroll (efficient retrieval)
        try:
            # Use scroll to get all points and filter by page_number
            scroll_result = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "page_number",
                            "match": {"value": page_number}
                        }
                    ]
                },
                limit=1
            )
            
            # Check if any results were found
            if scroll_result and len(scroll_result[0]) > 0:
                logger.info(f"✅ Page {page_number} already exists in collection '{collection_name}'")
                return True
            else:
                return False
                
        except Exception as scroll_error:
            logger.debug(f"Could not check page existence: {str(scroll_error)}")
            return False
            
    except Exception as e:
        logger.debug(f"Error checking page existence: {str(e)}")
        return False


def get_processed_pages(pdf_name: str) -> List[int]:
    """Get a list of all page numbers that have already been processed for a PDF.
    
    Args:
        pdf_name: Name of the PDF file
    
    Returns:
        List of page numbers that have been processed
    """
    if qdrant_client is None:
        return []
    
    try:
        # Create collection name from PDF name
        import re
        collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pdf_name.replace('.pdf', ''))
        collection_name = collection_name[:64]
        
        # Check if collection exists
        try:
            collections = qdrant_client.get_collections().collections
            collection_exists = any(col.name == collection_name for col in collections)
            
            if not collection_exists:
                return []
        except Exception:
            return []
        
        # Retrieve all points from the collection
        processed_pages = []
        offset = None
        
        while True:
            try:
                scroll_result = qdrant_client.scroll(
                    collection_name=collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = scroll_result
                
                for point in points:
                    page_num = point.payload.get("page_number")
                    if page_num is not None:
                        processed_pages.append(page_num)
                
                # Check if there are more results
                if next_offset is None:
                    break
                    
                offset = next_offset
                
            except Exception as scroll_error:
                logger.warning(f"⚠️ Error scrolling through collection: {str(scroll_error)}")
                break
        
        # Sort and return unique page numbers
        processed_pages = sorted(list(set(processed_pages)))
        logger.info(f"📊 Found {len(processed_pages)} already processed pages for '{pdf_name}'")
        return processed_pages
        
    except Exception as e:
        logger.warning(f"⚠️ Error getting processed pages: {str(e)}")
        return []


def search_qdrant(query: str, limit: int = 5) -> List[Dict]:
    """Search Qdrant for similar documents across ALL collections."""
    if not qdrant_client:
        st.error("Qdrant client is not initialized.")
        return []

    try:
        logger.info(f"🔍 Searching for: '{query[:100]}...'" )
        query_embedding = generate_embedding(query, task_type="retrieval_query")
        
        if query_embedding is None:
            logger.error("❌ Failed to generate query embedding")
            return []

        logger.info(f"✅ Query embedding generated (dimension: {len(query_embedding)})" )

        # Get all collections
        collections_response = qdrant_client.get_collections()
        all_collections = [col.name for col in collections_response.collections]
        logger.info(f"📂 Searching in {len(all_collections)} collections: {all_collections}")

        all_results = []
        for collection_name in all_collections:
            try:
                # Use the 'search' method, which is the correct method for recent versions
                search_results = qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit
                )
                
                for hit in search_results:
                    all_results.append({
                        "score": hit.score,
                        "page_number": hit.payload.get("page_number", "N/A"),
                        "pdf_name": hit.payload.get("pdf_name", collection_name),
                        "text": hit.payload.get("clean_text", "")
                    })
            except Exception as col_search_error:
                logger.warning(f"⚠️ Could not search in collection '{collection_name}': {col_search_error}")

        # Sort all results by score (highest first)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return the top results up to the limit
        top_results = all_results[:limit]
        
        logger.info(f"✅ Found {len(top_results)} results across all collections.")
        return top_results

    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}" )
        st.error(f"Search error: {str(e)}" )
        return []


def process_pdf(pdf_file, pdf_name: str, dpi: int = 300, use_gemini: bool = False) -> Dict:
    """Main function to process PDF: OCR, clean, embed, and store.
    
    Args:
        pdf_file: PDF file object
        pdf_name: Name of the PDF file
        dpi: DPI for image conversion (default: 300)
        use_gemini: Whether to use Gemini for text enhancement (slower but higher quality)
    """
    logger.info("="*80)
    logger.info(f"🚀 STARTING PDF PROCESSING: {pdf_name}")
    logger.info("="*80)
    
    results = {
        "total_pages": 0,
        "successful_pages": 0,
        "failed_pages": [],
        "page_texts": {}
    }
    
    # Save to temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getbuffer())
        tmp_path = tmp_file.name
    
    logger.info(f"✅ PDF loaded to temporary file: {tmp_path}")
    
    try:
        # Get total number of pages
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(tmp_path)
        total_pages = info["Pages"]
        results["total_pages"] = total_pages
        
        logger.info(f"📊 Total pages in PDF: {total_pages}")
        logger.info(f"⚙️ Settings - DPI: {dpi}")
        logger.info("-"*80)
        
        # ✨ RESUMABLE UPLOAD FEATURE: Check for already processed pages
        processed_pages = get_processed_pages(pdf_name)
        skipped_pages = 0
        resumed_pages_count = len(processed_pages)
        
        if processed_pages:
            logger.info("")
            logger.info("🔄"*40)
            logger.info("✨ RESUMABLE UPLOAD: Found previously processed pages!")
            logger.info(f"📊 Already processed: {len(processed_pages)} pages")
            logger.info(f"📋 Processed pages: {processed_pages}")
            logger.info(f"⏭️ These pages will be SKIPPED to resume from where upload stopped")
            logger.info("🔄"*40)
            logger.info("")
            st.info(f"🔄 **Resuming upload**: {len(processed_pages)} pages already processed. Continuing from where it stopped...")
            # Update results to reflect already processed pages
            results["successful_pages"] = len(processed_pages)
        
        # Collection will be created automatically per PDF in store_in_qdrant()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Batch processing configuration
        BATCH_SIZE = 10
        batch_start = 1
        
        # Process each page
        for page_num in range(1, total_pages + 1):
            # ✨ RESUMABLE UPLOAD: Skip already processed pages
            if page_num in processed_pages:
                logger.info(f"⏭️ SKIPPING Page {page_num}/{total_pages} - Already processed")
                skipped_pages += 1
                # Load existing text for this page from Qdrant to display in results
                try:
                    import re
                    collection_name = re.sub(r'[^a-zA-Z0-9_-]', '_', pdf_name.replace('.pdf', ''))
                    collection_name = collection_name[:64]
                    scroll_result = qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter={
                            "must": [
                                {
                                    "key": "page_number",
                                    "match": {"value": page_num}
                                }
                            ]
                        },
                        limit=1,
                        with_payload=True
                    )
                    if scroll_result and len(scroll_result[0]) > 0:
                        existing_text = scroll_result[0][0].payload.get("clean_text", "")
                        if existing_text:
                            results["page_texts"][page_num] = existing_text
                except Exception as e:
                    logger.debug(f"Could not retrieve existing text for page {page_num}: {str(e)}")
                
                # Update progress
                progress_bar.progress(page_num / total_pages)
                continue
            
            logger.info("")
            logger.info(f"{'='*80}")
            logger.info(f"📑 PROCESSING PAGE {page_num}/{total_pages}")
            logger.info(f"{'='*80}")
            
            status_text.text(f"Processing page {page_num}/{total_pages}...")
            
            try:
                # Step 1: Convert PDF page to image (ALWAYS use image-based OCR)
                logger.info(f"🖼️ Step 1/5: Converting page {page_num} to image for OCR...")
                image = pdf_page_to_image(tmp_path, page_num, dpi)
                
                if image is None:
                    results["failed_pages"].append({
                        "page": page_num,
                        "error": "Failed to convert page to image"
                    })
                    logger.error(f"❌ Page {page_num} failed: Could not convert to image")
                    continue
                
                # Step 2: Perform OCR on the image
                logger.info(f"🔍 Step 2/5: Performing OCR on page {page_num}...")
                raw_text = perform_ocr(image)
                
                # If OCR failed and we're still at default DPI, try with higher DPI
                if (not raw_text or raw_text.strip() == "") and dpi < 400:
                    logger.warning(f"⚠️ OCR failed at {dpi} DPI, retrying with higher DPI (400)...")
                    higher_dpi_image = pdf_page_to_image(tmp_path, page_num, 400)
                    if higher_dpi_image is not None:
                        raw_text = perform_ocr(higher_dpi_image)
                        if raw_text and raw_text.strip():
                            logger.info("✅ Higher DPI retry succeeded")
                
                if not raw_text or raw_text.strip() == "":
                    results["failed_pages"].append({
                        "page": page_num,
                        "error": "Page unreadable - no text detected"
                    })
                    logger.error(f"❌ Page {page_num} failed: No text detected")
                    continue
                
                # Step 3: Clean and correct text
                logger.info(f"🧹 Step 3/5: Cleaning and correcting text for page {page_num}...")
                logger.info(f"📝 Raw text length: {len(raw_text)} characters")
                logger.info(f"📝 Raw text preview: {raw_text[:200] if len(raw_text) > 200 else raw_text}")
                
                # First, do basic cleaning
                clean_text = clean_ocr_text(raw_text)
                logger.info(f"✅ Basic cleaning done - Length: {len(clean_text)} characters")
                
                # Then enhance with Gemini for Tamil accuracy (OPTIONAL for speed)
                if use_gemini and clean_text and clean_text.strip():
                    logger.info(f"🤖 Applying Gemini enhancement (this may take time)...")
                    enhanced_text = gemini_enhance_tamil_ocr(clean_text, use_enhanced_prompt=True)
                    # Only use enhanced text if it's not empty
                    if enhanced_text and enhanced_text.strip():
                        clean_text = enhanced_text
                        logger.info(f"✅ Gemini enhancement applied - Final length: {len(clean_text)} characters")
                    else:
                        logger.warning(f"⚠️ Gemini enhancement returned empty text, using basic cleaned text")
                elif not use_gemini:
                    logger.info(f"⚡ Skipping Gemini enhancement for SPEED")
                
                if clean_text:
                    logger.info(f"📝 Final text preview: {clean_text[:200] if len(clean_text) > 200 else clean_text}")
                
                if not clean_text or clean_text.strip() == "":
                    results["failed_pages"].append({
                        "page": page_num,
                        "error": "Page unreadable - no valid text after cleaning"
                    })
                    logger.error(f"❌ Page {page_num} failed: No valid text after cleaning")
                    continue
                
                # Step 4: Generate embedding
                logger.info(f"🧠 Step 4/5: Generating embedding for page {page_num}...")
                embedding = generate_embedding(clean_text)
                
                if embedding is None:
                    results["failed_pages"].append({
                        "page": page_num,
                        "error": "Failed to generate embedding"
                    })
                    logger.error(f"❌ Page {page_num} failed: Could not generate embedding")
                    continue
                
                # Step 5: Store in Qdrant
                logger.info(f"💾 Step 5/5: Storing page {page_num} in Qdrant...")
                success = store_in_qdrant(pdf_name, page_num, clean_text, embedding)
                
                if success:
                    results["successful_pages"] += 1
                    results["page_texts"][page_num] = clean_text
                    logger.info(f"✅ Page {page_num} completed successfully!")
                else:
                    results["failed_pages"].append({
                        "page": page_num,
                        "error": "Failed to store in Qdrant"
                    })
                    logger.error(f"❌ Page {page_num} failed: Could not store in Qdrant")
                
            except Exception as e:
                results["failed_pages"].append({
                    "page": page_num,
                    "error": str(e)
                })
                logger.error(f"❌ Page {page_num} failed with exception: {str(e)}")
            
            # Update progress
            progress_bar.progress(page_num / total_pages)
            
            # Batch completion logging
            if page_num % BATCH_SIZE == 0 or page_num == total_pages:
                batch_end = page_num
                successful_in_batch = sum(1 for p in range(batch_start, batch_end + 1) 
                                         if p not in [fp["page"] for fp in results["failed_pages"]])
                logger.info("")
                logger.info(f"{'🎯'*40}")
                logger.info(f"✅ BATCH COMPLETED: Pages {batch_start} to {batch_end}")
                logger.info(f"   • Successful: {successful_in_batch}/{batch_end - batch_start + 1}")
                logger.info(f"   • Total Progress: {page_num}/{total_pages} pages ({(page_num/total_pages*100):.1f}%)")
                logger.info(f"{'🎯'*40}")
                logger.info("")
                batch_start = batch_end + 1
        
        status_text.text("Processing complete!")
        
        # Final summary
        logger.info("")
        logger.info("="*80)
        logger.info("🎉 PDF PROCESSING COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Summary:")
        logger.info(f"   • Total Pages: {results['total_pages']}")
        logger.info(f"   • Successful: {results['successful_pages']}")
        if skipped_pages > 0:
            logger.info(f"   • Skipped (Already Processed): {skipped_pages}")
            logger.info(f"   • Newly Processed: {results['successful_pages'] - resumed_pages_count}")
        logger.info(f"   • Failed: {len(results['failed_pages'])}")
        logger.info(f"   • Success Rate: {(results['successful_pages']/results['total_pages']*100):.1f}%")
        
        if results['failed_pages']:
            logger.warning(f"⚠️ Failed pages:")
            for failure in results['failed_pages']:
                logger.warning(f"   • Page {failure['page']}: {failure['error']}")
        
        logger.info(f"✅ All {results['successful_pages']} pages stored in Qdrant (Collection: {pdf_name.replace('.pdf', '')})")
        logger.info("="*80)
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)
    
    return results


def export_raw_ocr_text(pdf_file, pdf_name: str, dpi: int = 300) -> Dict:
    """Extract raw OCR text without any cleaning or enhancement."""
    logger.info("="*80)
    logger.info(f"🚀 EXTRACTING RAW TEXT: {pdf_name}")
    logger.info("="*80)
    
    results = {
        "total_pages": 0,
        "extracted_pages": 0,
        "page_texts": {}
    }
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(tmp_path)
        total_pages = info["Pages"]
        results["total_pages"] = total_pages
        
        logger.info(f"📊 Total pages in PDF: {total_pages}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page_num in range(1, total_pages + 1):
            status_text.text(f"Extracting page {page_num}/{total_pages}...")
            
            try:
                # Convert page to image
                image = pdf_page_to_image(tmp_path, page_num, dpi)
                if image is None:
                    logger.warning(f"⚠️ Page {page_num}: Could not convert to image")
                    continue
                
                # Perform OCR - NO CLEANING, NO ENHANCEMENT
                raw_text = perform_ocr(image)
                
                if raw_text and raw_text.strip():
                    results["page_texts"][page_num] = raw_text
                    results["extracted_pages"] += 1
                    logger.info(f"✅ Page {page_num}: {len(raw_text)} characters extracted")
                else:
                    logger.warning(f"⚠️ Page {page_num}: No text detected")
                
            except Exception as e:
                logger.error(f"❌ Page {page_num}: {str(e)}")
            
            progress_bar.progress(page_num / total_pages)
        
        status_text.text("Extraction complete!")
        
    finally:
        os.unlink(tmp_path)
    
    return results


def list_available_gemini_models():
    """List all available Gemini models for debugging."""
    try:
        logger.info("📋 Listing available Gemini models...")
        models = genai.list_models()
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                # Strip 'models/' prefix to get clean model name
                model_name = model.name.replace('models/', '')
                available.append(model_name)
                logger.info(f"   ✅ {model_name}")
        return available
    except Exception as e:
        logger.error(f"❌ Error listing models: {str(e)}")
        return []


def generate_rag_response(query: str, context_docs: List[Dict], query_language: str = "en") -> str:
    """Generate answer using Gemini with retrieved context."""
    try:
        if not context_docs:
            return "I couldn't find any relevant information in the documents to answer your question."
        
        # Prepare context
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            context_text += f"Source {i} (File: {doc['pdf_name']}, Page: {doc['page_number']}):\n{doc['text']}\n\n"
            
        # Construct prompt
        prompt = f"""
        You are a helpful AI assistant capable of answering questions based on the provided document context.
        
        Context:
        {context_text}
        
        Question: {query}
        
        Instructions:
        1. Answer the question specifically using ONLY the provided context.
        2. The user's query is in { 'Tamil' if query_language == 'ta' else 'English' }. YOU MUST ANSWER IN THE SAME LANGUAGE.
        3. If the answer is not in the context, respond in the user's language with "I cannot find the answer in the provided documents."
        4. Cite your sources clearly. For every fact, mention the Source number, File name, and Page number.
        5. Format the answer nicely with markdown.
        6. Be concise but complete.
        
        Answer:
        """
        
        # First, try to get list of available models
        available_models = list_available_gemini_models()
        
        if not available_models:
            logger.warning("⚠️ Could not list models, trying default models...")
            # Extended list of possible model names to try (without models/ prefix)
            available_models = [
                'gemini-1.5-flash-latest',
                'gemini-1.5-pro-latest',
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro',
                'gemini-1.5-flash-002',
                'gemini-1.5-pro-002'
            ]
        
        response = None
        last_error = None
        
        for model_name in available_models:
            try:
                logger.info(f"🤖 Trying model: {model_name}")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                logger.info(f"✅ Successfully used model: {model_name}")
                return response.text
            except Exception as model_error:
                last_error = model_error
                logger.warning(f"⚠️ Model {model_name} failed: {str(model_error)}")
                continue
        
        # If all models failed, return helpful error
        error_msg = f"""
        ❌ Unable to connect to Gemini API. All models failed.
        
        **Possible solutions:**
        1. Check if your API key is valid: https://aistudio.google.com/app/apikey
        2. Verify your API key has access to Gemini models
        3. Try regenerating your API key
        4. Check if there are any billing issues with your Google Cloud account
        
        Last error: {str(last_error)}
        """
        return error_msg
        
    except Exception as e:
        logger.error(f"❌ LLM generation error: {str(e)}")
        return f"Error generating answer: {str(e)}\n\nPlease check your GEMINI_API_KEY configuration."
        


# Streamlit UI
def main():
    st.title("📄 PaddleOCR + RAG Processing System")
    st.markdown("*Using PaddleOCR as the primary OCR engine for Tamil & English document processing*")
    st.markdown("---")
    
    # Settings in sidebar
    with st.sidebar:
        st.header("Navigation")
        
        # Tab selection in sidebar
        selected_tab = st.radio(
            "Choose an option:",
            ["📤 Upload & Process", "🔍 Search & Retrieve"],
            label_visibility="collapsed",
            key="main_nav"
        )
    
    # Speed settings - Gemini enhancement ENABLED for ACCURATE text matching
    use_gemini_enhancement = True  # AI corrects OCR errors to match original image
    
    # Default OCR settings (used internally)
    dpi = 300
    
    # Main content based on selected tab
    if selected_tab == "📤 Upload & Process":
        st.header("Upload PDF for OCR Processing")
        
        # Resumable upload feature info
        st.info(
            "✨ **Resumable Uploads Enabled!** If a PDF upload is interrupted or stopped, "
            "simply re-upload the same PDF file and processing will automatically continue "
            "from where it stopped. Already processed pages will be skipped."
        )
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to process",
            key="pdf_uploader"
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} file(s)")
            
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                for uploaded_file in uploaded_files:
                    st.markdown(f"### 📄 Processing: {uploaded_file.name}")
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        results = process_pdf(
                            uploaded_file,
                            uploaded_file.name,
                            dpi,
                            use_gemini=use_gemini_enhancement
                        )
                        
                        # Display results in an expander for each file
                        with st.expander(f"✅ Results for {uploaded_file.name}", expanded=True):
                            st.subheader("📊 Processing Stats")
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total Pages", results["total_pages"])
                            with col_b:
                                st.metric("Successful", results["successful_pages"], 
                                         delta=f"{(results['successful_pages']/results['total_pages']*100):.1f}%")
                            with col_c:
                                st.metric("Failed", len(results["failed_pages"]))
                            
                            if results["failed_pages"]:
                                st.warning("Some pages failed to process:")
                                for failure in results["failed_pages"]:
                                    st.write(f"- Page {failure['page']}: {failure['error']}")
                            
                            # Display processed text for each page
                            if results["page_texts"]:
                                st.markdown("---")
                                st.subheader("📝 Processed Text by Page")
                                st.info("✅ Each page has been cleaned, enhanced, and stored in Qdrant")
                            
                            for page_num in sorted(results["page_texts"].keys()):
                                with st.expander(f"📄 Page {page_num}", expanded=(page_num == 1)):
                                    text = results["page_texts"][page_num]
                                    st.text_area(
                                        f"Page {page_num} - Processed Text",
                                        value=text,
                                        height=300,
                                        disabled=True,
                                        key=f"processed_page_{uploaded_file.name}_{page_num}"
                                    )


    elif selected_tab == "🔍 Search & Retrieve":
        st.header("Chat with your Documents")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Detect if query is in Tamil
            query_language = detect_language(prompt)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt, "language": query_language})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Searching for relevant documents..."):
                    context_docs = search_qdrant(prompt)
                    # The response is now generated directly in the correct language
                    response = generate_rag_response(prompt, context_docs, query_language=query_language)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response, "language": query_language})

if __name__ == "__main__":
    main()
