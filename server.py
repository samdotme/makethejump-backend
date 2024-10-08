import os
from dotenv import load_dotenv
import json
from llm_logic import LlmLogicBrain

# Set the cache directory to /tmp
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['HF_HOME'] = '/tmp'

class BrainHandler:
    brain = None

    @classmethod
    def initialize_brain(cls):
        # Load environment variables
        load_dotenv()
        
        pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')
        
        cls.brain = LlmLogicBrain(pinecone_index_name)

    @classmethod
    def get_response(cls, prompt):
        if cls.brain is None:
            raise ValueError("Brain not initialized")
        return cls.brain.respond_with_chain(prompt)


def lambda_handler(event, context):    
    # Initialize the brain once at the start
    BrainHandler.initialize_brain()
    
    try:
        # Check if brain is initialized
        if BrainHandler.brain is None:
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS,PUT,DELETE',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Credentials': 'true'
                },
                'body': json.dumps({'error': 'Brain not initialized'})
            }

        # Parse the path and query parameters from the event
        path = event.get('rawPath', '')
        if path == '/makethejump/bot':
            query_params = event.get('queryStringParameters', {})
            prompt = query_params.get('prompt', '')

            if not prompt:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS,PUT,DELETE',
                        'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                        'Access-Control-Allow-Credentials': 'true'
                    },
                    'body': json.dumps({'error': "'prompt' query parameter is required"})
                }

            response = BrainHandler.get_response(prompt)
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS,PUT,DELETE',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Credentials': 'true'
                },
                'body': json.dumps({'response': response})
            }
        else:
            return {
                'statusCode': 404,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS,PUT,DELETE',
                    'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                    'Access-Control-Allow-Credentials': 'true'
                },
                'body': json.dumps({'error': 'Not found'})
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET,POST,OPTIONS,PUT,DELETE',
                'Access-Control-Allow-Headers': 'Content-Type,Authorization',
                'Access-Control-Allow-Credentials': 'true'
            },
            'body': json.dumps({'error': str(e)})
        }

if __name__ == '__main__':
    # This block will not be executed in the Lambda environment
    # Add code here if you want to run the function locally for testing
    pass
