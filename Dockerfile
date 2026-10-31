# GRASP-SLM application image (Hugging Face Spaces Docker runtime, Render, or any Docker host).
#
#   docker build -t grasp-slm .
#   docker run -p 7860:7860 grasp-slm
#
# By default the image bundles the base Qwen2.5-1.5B GGUF. After training, build it with your model:
#   docker build -t grasp-slm --build-arg LLM_REPO=<you>/grasp-slm-gguf \
#                --build-arg LLM_FILE=grasp-slm-dpo-q4_k_m.gguf --build-arg ONNX_REPO=<you>/grasp-slm-onnx .

# ---- 1. Build the Spring Boot jar --------------------------------------------------------------
FROM eclipse-temurin:21-jdk AS build
WORKDIR /src
COPY shared shared
COPY app/.mvn app/.mvn
COPY app/mvnw app/pom.xml app/
RUN cd app && chmod +x mvnw && ./mvnw -q -B dependency:go-offline
COPY app/src app/src
RUN cd app && ./mvnw -q -B -DskipTests package

# ---- 2. Fetch model files ----------------------------------------------------------------------
FROM python:3.11-slim AS models
RUN pip install --no-cache-dir "huggingface_hub>=0.25"
COPY scripts/download_models.py /scripts/download_models.py
ARG LLM_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF
ARG LLM_FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf
ARG ONNX_REPO=
RUN python /scripts/download_models.py --models-dir /models --llm-repo "$LLM_REPO" --llm-file "$LLM_FILE" \
        ${ONNX_REPO:+--onnx-repo "$ONNX_REPO"}

# ---- 3. Runtime --------------------------------------------------------------------------------
FROM eclipse-temurin:21-jre
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/app/target/grasp-slm.jar app.jar
COPY --from=models /models models
# Spaces run the container as uid 1000; native libraries are unpacked into writable /tmp.
ENV PORT=7860 HOME=/tmp DJL_CACHE_DIR=/tmp/djl JAVA_TOOL_OPTIONS="-XX:MaxRAMPercentage=40"
RUN chown -R 1000 /app
USER 1000
EXPOSE 7860
CMD ["java", "-jar", "app.jar"]
