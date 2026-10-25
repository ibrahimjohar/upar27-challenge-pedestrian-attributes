#same torch / cuda as the Codabench evaluation container
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

RUN pip install --no-cache-dir timm==1.0.15 "numpy<2" pillow pandas matplotlib

WORKDIR /workspace
COPY src/ src/
COPY tools/ tools/
COPY submission/ submission/
COPY results/ results/

CMD ["bash"]
