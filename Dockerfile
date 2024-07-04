FROM public.ecr.aws/lambda/python:3.12

# Copy function code and dependencies
COPY requirements.txt  .
RUN pip install -r requirements.txt
COPY server.py  .
COPY hf_logic.py  .

# Set the CMD to start up the server
CMD ["python", "server.py"]
