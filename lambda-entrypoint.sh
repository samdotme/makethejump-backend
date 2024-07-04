#!/bin/sh

# Execute the Lambda function handler
exec python3 -m awslambdaric server.lambda_handler
