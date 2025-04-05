from transformers import pipeline

_qa_pipeline = None

def load_model(model_name="distilbert-base-cased-distilled-squad"):
    """
    Loads a transformer-based question-answering model.
    By default, uses a distilled BERT model fine-tuned on SQuAD.
    """
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = pipeline("question-answering", model=model_name)
    return _qa_pipeline

def answer_question(question, context):
    """
    Uses the QA model to answer the given question based on the provided context.
    Returns the result dictionary with answer, score, and positions.
    """
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = load_model()
    result = _qa_pipeline(question=question, context=context)
    return result
