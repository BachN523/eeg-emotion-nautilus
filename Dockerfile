# Image for running folds as Kubernetes Jobs on Nautilus.
# Build and push to the NRP GitLab container registry, then point the image:
# field in k8s/loso-job.yaml at it instead of cloning the repo into the PVC.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
ENTRYPOINT ["python", "main.py"]
