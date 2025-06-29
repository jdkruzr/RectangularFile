# Rectangular File

![image](https://github.com/user-attachments/assets/c168eea6-1992-473e-9427-39dbf52b12c9)

Rectangular File is a document management and analysis system originally intended for use with Onyx Boox e-ink tablets, to enhance your ability to search for text in your handwriting and build an index of recognized text for later use. Supernote does this with their handwriting search functionality, but Boox devices start over every time you start a new search. This, how you say, Le Sucks.

RF is a Flask application which uses a locally deployed version of the Qwen2-VL-2B LLM to analyze and transcribe the contents of your handwritten notes. You deploy it on a server somewhere, tell your tablet devices to dump their data into a folder on the server -- on Boox this is just an auto-PDF-upload feature -- and RF will ingest them, perform text extraction and handwriting recognition, and then add the data to its searchable index, then show you where it got the search results from when it finds what you're looking for.

Eventually we'd also like to add features like wordclouds, etc.

Basically, think "what if Paperless-ngx, but it didn't vomit when asked to look at handwriting?"

You could conceivably use this with notes from any device -- Kindle Scribe, reMarkable, etc.

## Deployment Details

For authentication, you'll want to add these environment variables to your systemd unit file.

**Generate a secure secret key**

python -c 'import secrets; print(secrets.token_hex(32))'

Put the above into your unit file with ENVIRONMENT=SECRET_KEY=[output of the above]

**Generate password hash (replace 'yourpassword' with your actual password)**

python -c "import hashlib; print(hashlib.sha256('yourpassword'.encode()).hexdigest())"

Put the above into your unit file with ENVIRONMENT=APP_PASSWORD_HASH=[output of the above]

## System Dependencies

Before installing the Python requirements, you need to install these system packages:

### macOS (using Homebrew):
```bash
brew install poppler  # Required for PDF processing
```

### Debian/Ubuntu:
```bash
apt-get update
apt-get install -y python3-pip python3-venv  # Required for PDF processing and server
```

### Windows:
Not supported.
