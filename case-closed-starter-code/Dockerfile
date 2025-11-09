# Participant testing Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copy participant code
COPY agent.py .
COPY ppo_policy_update_100.pth .
COPY requirements.txt .
COPY model.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD ["python", "agent.py"]
