FROM python:3.9.5
WORKDIR /app
COPY . .
# RUN pip install -r requirements.txt 
RUN pip install numpy scipy
RUN pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html