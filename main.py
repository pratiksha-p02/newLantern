from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging

# ----------------- Setup -----------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

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
    if "MRI" in text or "MAGNETIC RESONANCE" in text:
        return "MRI"
    if "CT" in text or "COMPUTED TOMOGRAPHY" in text or "CAT SCAN" in text:
        return "CT"
    if "XRAY" in text or "X-RAY" in text or "PLAIN FILM" in text or "ROENTGEN" in text:
        return "XRAY"
    if "ULTRASOUND" in text or "US" in text or "SONO" in text or "ECHO" in text:
        return "US"
    if "PET" in text or "POSITRON EMISSION" in text:
        return "PET"
    return "OTHER"

def extract_body_part(text: str):
    text = text.upper()
    if "BRAIN" in text or "HEAD" in text or "NEURO" in text or "CEREBRAL" in text or "CRANIAL" in text or "INTRACRANIAL" in text:
        return "BRAIN"
    if "CHEST" in text or "THORAX" in text or "THORACIC" in text or "LUNG" in text or "PULMONARY" in text:
        return "CHEST"
    if "ABDOMEN" in text or "ABDOMINAL" in text or "GASTROINTESTINAL" in text or "GI" in text or "LIVER" in text or "KIDNEY" in text:
        return "ABDOMEN"
    if "SPINE" in text or "VERTEBRAL" in text or "SPINAL" in text:
        return "SPINE"
    if "PELVIS" in text or "PELVIC" in text or "BLADDER" in text or "PROSTATE" in text:
        return "PELVIS"
    return "OTHER"

def is_relevant(current: Study, prior: Study) -> bool:
    curr_text = current.study_description
    prior_text = prior.study_description

    # Rule-based signals
    curr_mod = extract_modality(curr_text)
    prior_mod = extract_modality(prior_text)

    curr_body = extract_body_part(curr_text)
    prior_body = extract_body_part(prior_text)

    # Strong rule match: same modality or same body part
    if curr_mod == prior_mod and curr_mod != "OTHER":
        return True
    if curr_body == prior_body and curr_body != "OTHER":
        return True

    # Additional simple checks: if descriptions share key terms
    curr_words = set(curr_text.upper().split())
    prior_words = set(prior_text.upper().split())
    common_words = curr_words & prior_words
    if len(common_words) > 1:  # lowered threshold
        return True

    return False

# ----------------- Endpoint -----------------
@app.post("/predict")
def predict(request: RequestModel):
    try:
        predictions = []

        logging.info(f"Received {len(request.cases)} cases")

        total_priors = sum(len(c.prior_studies) for c in request.cases)
        logging.info(f"Total priors: {total_priors}")

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
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return {"error": "Internal server error", "details": str(e)}
