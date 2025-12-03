1. Prerequisites
Python 3.9 or 3.10 installed
Windows, macOS, or Linux
A webcam connected and accessible by the system
Git installed 


2. Download the project
git clone <PRIVATE_REPO_URL>
cd ml-fit-trainer


3. Create and activate a virtual environment
Run these commands inside the project folder.
Windows (PowerShell/CMD/VsCode):
python -m venv venv
venv\Scripts\activate
#MacOS/Linux
python3 -m venv venv
source venv/bin/activate


4.Install dependencies
In the activated virtual environment:
pip install --upgrade pip
pip install -r requirements.txt

5.Run the app
streamlit run web_app.py
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Open the URL if it does not open automatically.

6.Allow camera access
When the Streamlit page loads:
Your browser may ask for camera permission. Select "Allow".
Enable the "Start Camera" checkbox or button inside the UI.
The webcam feed will appear along with live pose detection and counters.

7.Stopping the app
Ctrl+C in Terminal
