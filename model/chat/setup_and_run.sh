# #!/bin/bash

# # Navigate to the model/chat directory
# # cd model/chat || exit
# # Check if the script is run from the correct directory
# if [ ! -d "model/chat" ]; then
#   echo "This script must be run from the model/chat directory."
#   exit 1
# fi
# cd model/chat
# Clear the chat_history.json file (if it exists)
if [ -f chat_history.json ]; then
  > chat_history.json  # This will empty the file
fi
# # Check if Python 3 is installed
# if ! command -v python3 &> /dev/null; then
#     echo "Python 3 is not installed. Please install Python 3."
#     exit 1
# fi
# Start the Flask app in the background
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py &

# Get the Flask server's PID (Process ID)
FLASK_PID=$!

# Wait for the user to press ENTER to stop the Flask server
read -p "Press ENTER to stop the Flask server and clean up..."

# Kill the Flask process
kill $FLASK_PID

# Deactivate the virtual environment
deactivate

# Delete the virtual environment folder
rm -rf venv

# Delete Python cache and __pycache__ folders
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -exec rm -f {} +

# Delete everything in the static folder
rm -rf static/*

# Inform the user that cleanup is complete
echo "Flask server stopped, virtual environment deactivated, and cache cleaned up."
echo "Static folder cleared and chat history file emptied."