<h1 align="center">The Cat Cafe</h1>

<p align="center">
  This is the back end for an intelligent chatbot that powers a fictional Cat Cafe adoption service. It's backed by an LLM that uses your proprietary data and keeps it private.
</p>

<p align="center">
  <strong>Build your own at <a href="http://makethejump.ai/developer">MakeTheJump.ai/developer</a></strong>
</p>
<br/>

## Getting Started

### Install Python

I'm recording several video to walk you through this process, both on a Mac and a Windows computer.

Mac: [Link pending]

Windows: [Link pending]

### Set Up Environment

Copy the file `.env.example` and paste. Then rename the pasted file to be `.env`.

Fill out the environment variables with your values.

### Install Dependencies

From a root prompt type:

    pip install -r requirements.txt

This will take a few minutes to download, even on a fast internet connection.

### Running Locally

From a command prompt run

    python index.py

You should see an output of text that ends with:

    Loaded embeddings
    Loaded vector store
    Starting server on port 8000

### Testing

You can do a quick check of the system by visiting http://localhost:8000/makethejump/bot?prompt=suggest+a+cat+breed+for+me

To test it with a front end user interface, you'll want to fire up the companion front end repository. You can find that here: https://github.com/samdotme/makethejump-bot
