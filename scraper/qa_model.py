from transformers import pipeline
import torch
import logging
import os
from .field_mapper import field_mapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_qa_pipeline = None

def load_model():
    global _qa_pipeline
    if _qa_pipeline is not None:
        return _qa_pipeline

    try:
        # Force CUDA if available
        if torch.cuda.is_available():
            # Set CUDA device
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            device = torch.device('cuda:0')
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, falling back to CPU")

        # Load the QA pipeline with CUDA if available
        _qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        
        # Verify device
        logger.info(f"Model loaded on device: {next(_qa_pipeline.model.parameters()).device}")
        return _qa_pipeline
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def answer_question(question: str, context: str) -> dict:
    """
    Uses the QA model to answer the given question based on the provided context.
    Returns the result dictionary with answer, score, and positions.
    """
    try:
        if _qa_pipeline is None:
            _qa_pipeline = load_model()

        # Enhance the question with better context understanding
        enhanced_question = field_mapper.enhance_question(question)
        logger.info(f"Original question: {question}")
        logger.info(f"Enhanced question: {enhanced_question}")

        # Get relevant fields based on context
        relevant_fields = field_mapper.understand_context(question)
        logger.info(f"Relevant fields detected: {relevant_fields}")

        # Special handling for industry/domain questions
        if 'domain' in relevant_fields and 'industry' in question.lower():
            industry_questions = [
                "What is the industry or field of activity of this organization?",
                "What sector does this organization operate in?",
                "What is the main business area or specialty of this organization?",
                "What is the organization's domain of expertise?"
            ]
            best_result = None
            for q in industry_questions:
                result = _qa_pipeline(question=q, context=context)
                if result['score'] > 0.3 and (not best_result or result['score'] > best_result['score']):
                    best_result = result
            if best_result:
                return best_result

        # Get the answer from the QA model
        result = _qa_pipeline(question=enhanced_question, context=context)
        
        # If the answer is empty or score is low, try with the original question
        if not result['answer'] or result['score'] < 0.3:
            logger.info("Low confidence answer, trying with original question")
            result = _qa_pipeline(question=question, context=context)

        # If we have relevant fields, try to validate the answer
        if relevant_fields and result['answer']:
            # Check if the answer matches any of the expected field patterns
            for field in relevant_fields:
                mapped_field = field_mapper.map_field(result['answer'])
                if mapped_field and mapped_field in relevant_fields:
                    logger.info(f"Answer validated as {mapped_field}")
                    break

        return result
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return {"answer": "", "score": 0.0, "start": 0, "end": 0}

def extract_fields(text: str, context: str) -> dict:
    """
    Extract all relevant fields from the given text using the QA model.
    """
    fields = {}
    relevant_fields = field_mapper.understand_context(text)
    
    for field in relevant_fields:
        if field == "domain":
            # Special handling for industry/domain field
            questions = [
                "What is the industry or field of activity?",
                "What sector does this organization operate in?",
                "What is the main business area or specialty?",
                "What is the organization's domain of expertise?"
            ]
            best_result = None
            for question in questions:
                result = answer_question(question, context)
                if result['answer'] and result['score'] > 0.3:
                    if not best_result or result['score'] > best_result['score']:
                        best_result = result
            if best_result:
                fields[field] = best_result['answer']
        else:
            question = f"What is the {field}?"
            result = answer_question(question, context)
            if result['answer'] and result['score'] > 0.3:
                fields[field] = result['answer']
    
    return fields
