FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["diffneoscore"]
