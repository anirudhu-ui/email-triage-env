FROM python:3.10-slim

WORKDIR /app

# Copy all project files
COPY . .
ENV ENABLE_WEB_INTERFACE=true
# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic requests openai anthropic colorama

# Expose the standard HF Spaces port
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
