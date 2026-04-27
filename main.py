from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util
import logging

# ----------------- Setup -----------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

model = SentenceTransformer("all-MiniLM-L6-v2")

# embedding cache
embedding_cache = {}

# ----------------- Models -----------------
class Study(BaseModel):
    study_id: str
    study_description: str
    study_date: str

class Case(BaseModel):
    case_id: str
    patient_id: str
    patient_name: str
    current_study: Study
    prior_studies: List[Study]

class RequestModel(BaseModel):
    challenge_id: str
    schema_version: int
    generated_at: str
    cases: List[Case]

# ----------------- Helpers -----------------
def extract_modality(text: str):
    text = text.upper()
    if "MRI" in text:
        return "MRI"
    if "CT" in text:
        return "CT"
    if "XRAY" in text or "X-RAY" in text:
        return "XRAY"
    return "OTHER"

def extract_body_part(text: str):
    text = text.upper()
    if "BRAIN" in text or "HEAD" in text:
        return "BRAIN"
    if "CHEST" in text:
        return "CHEST"
    if "ABDOMEN" in text:
        return "ABDOMEN"
    return "OTHER"

def get_embedding(text: str):
    return embedding_cache[text]

def is_relevant(current: Study, prior: Study) -> bool:
    curr_text = current.study_description
    prior_text = prior.study_description

    # rule signals
    curr_mod = extract_modality(curr_text)
    prior_mod = extract_modality(prior_text)

    curr_body = extract_body_part(curr_text)
    prior_body = extract_body_part(prior_text)

    # strong rule match
    if curr_mod == prior_mod and curr_body == prior_body:
        return True

    # semantic similarity
    emb1 = get_embedding(curr_text)
    emb2 = get_embedding(prior_text)
    sim_score = util.cos_sim(emb1, emb2).item()

    if sim_score > 0.7:
        return True

    return False

# ----------------- Endpoint -----------------
@app.post("/predict")
def predict(request: RequestModel):
    predictions = []

    logging.info(f"Received {len(request.cases)} cases")

    total_priors = sum(len(c.prior_studies) for c in request.cases)
    logging.info(f"Total priors: {total_priors}")

    # -------- Collect all unique texts --------
    all_texts = set()

    for case in request.cases:
        all_texts.add(case.current_study.study_description)
        for prior in case.prior_studies:
            all_texts.add(prior.study_description)

    # -------- Batch encode (FAST) --------
    new_texts = [t for t in all_texts if t not in embedding_cache]

    if new_texts:
        embeddings = model.encode(new_texts, batch_size=64)

        for text, emb in zip(new_texts, embeddings):
            embedding_cache[text] = emb

    # -------- Prediction loop --------
    for case in request.cases:
        for prior in case.prior_studies:
            predictions.append({
                "case_id": case.case_id,
                "study_id": prior.study_id,
                "predicted_is_relevant": is_relevant(
                    case.current_study, prior
                )
            })

    return {"predictions": predictions}