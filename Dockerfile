FROM public.ecr.aws/lambda/python:3.12

# Copy function code and dependencies
COPY requirements.txt  .
RUN pip install -r requirements.txt
COPY server.py  .
COPY hf_logic.py  .
COPY lambda-entrypoint.sh /lambda-entrypoint.sh
RUN chmod +x /lambda-entrypoint.sh

# Set the entrypoint to use the Lambda Runtime Interface Client
ENTRYPOINT ["/lambda-entrypoint.sh"]
