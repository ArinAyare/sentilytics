# Documentation plan (generate next, once you have real metrics)
**Report chapters:** 1 Introduction · 2 Literature survey · 3 Problem statement & objectives · 4 Requirements (SRS) ·
5 System design (architecture, DFD, ER/collections, sequence diagrams) · 6 ML methodology · 7 Azure ML implementation ·
8 Backend/Frontend implementation · 9 Testing · 10 Results & discussion (baseline vs DistilBERT) · 11 Security ·
12 Conclusion & future scope · References.

**Screenshots to capture:** Azure ML job + metrics + registered model + endpoint (Studio), Swagger UI, each UI page,
confusion matrices, test run.

**Likely viva questions:** Why TF-IDF baseline *and* transformer? Why macro-F1 over accuracy? How is the neutral class handled?
What is MLflow doing? Difference between job, model registry, endpoint? What happens if Azure is down? How do you prevent
train/serve skew? (shared `ml/common.py`) · Cost control? (scale-to-zero cluster, delete endpoint) · Limitations? (sarcasm, domain shift, English only).
