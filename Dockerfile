# Use Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Train model during image build so `model.pkl` exists at runtime
RUN python train_model.py

# Expose port (Gradio usually runs on 7860, Flask on 5000)
EXPOSE 5000

# Run your app (change filename if needed)
CMD ["python", "app.py"]
