# Experiments

## Baseline
Started with sentence-transformers embeddings for semantic similarity matching between current and prior study descriptions.

## What worked
- Embeddings captured semantic relationships well
- Good accuracy on similar studies

## What failed
- High memory usage (200MB+ model)
- Slow startup times
- Timed out on Render free tier (512MB RAM)
- Not suitable for production deployment constraints

## Improvements Made
- Replaced embeddings with lightweight rule-based matching
- Extract modality (MRI, CT, XRAY, US, PET) and body part (BRAIN, CHEST, ABDOMEN, SPINE, PELVIS)
- Added word overlap check for additional relevance
- Reduced dependencies to fastapi + uvicorn only
- Memory footprint: ~50MB vs 200MB+
- Response time: <1s vs 10-30s

## Next Steps
- Expand medical terminology coverage
- Add temporal weighting (prioritize recent studies)
- Implement caching for repeated requests
- Consider fine-tuned lightweight models if memory allows
- Add more sophisticated text preprocessing (stemming, synonyms)
