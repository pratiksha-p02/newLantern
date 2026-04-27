# Relevant Priors API

This project exposes an HTTP API that determines whether prior radiology studies are relevant to a current study for a patient. The goal is to help radiologists focus on the most useful historical exams while reading new imaging studies.

---

##  Endpoint

**POST** `/predict`

### Request

Accepts a JSON payload containing:

* One or more cases
* Each case has:

  * A current study
  * A list of prior studies

### Response

Returns one prediction per prior study:

```json
{
  "predictions": [
    {
      "case_id": "string",
      "study_id": "string",
      "predicted_is_relevant": true
    }
  ]
}
```

---

##  Approach

This solution uses a **rule-based method**:

### Rule Based Signals

* Modality matching (MRI, CT, XRAY, US, PET)
* Body part matching (BRAIN, CHEST, ABDOMEN, SPINE, PELVIS)
* Word overlap check for additional relevance

### Decision Logic

* Match on modality + body part → relevant
* Otherwise, check word overlap threshold

---

## ⚡ Performance Optimizations

* Lightweight rule-based matching with no ML models
* Fast string processing and keyword extraction
* Minimal memory footprint (~50MB)
* No external dependencies beyond FastAPI
* Sub-second response times

---

##  Local Development

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run server

```bash
uvicorn main:app
```

### Open API docs

```
http://127.0.0.1:8000/docs
```

---

##  Deployment

Designed to run on platforms like Render using:

```bash
uvicorn main:app --host 0.0.0.0 --port 10000
```

---

##  Evaluation

* Each `(case_id, study_id)` pair is predicted independently
* Accuracy is computed as:

  ```
  correct_predictions / total_predictions
  ```
* Missing predictions count as incorrect

---

##  Future Improvements

* Incorporate temporal weighting (recency of studies)
* Use medical ontology for better anatomy matching
* Fine tune embeddings on radiology specific data
* Add lightweight classification layer on top of embeddings

---

##  Project Structure

```
main.py
requirements.txt
experiments.md
README.md
```

---

##  Summary

This solution balances:

* Accuracy via hybrid reasoning
* Speed via batching and caching
* Simplicity for reliable deployment

---
