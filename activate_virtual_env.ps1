py -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
$env:NO_MKDOCS_2_WARNING = 1
Write-Host "Starting mkdocs serve"
Write-Host "Don't forget to deploy GitHub"
Write-Host "mkdocs serve -a 127.0.0.1:8001"
cd dive-into-deep-learning
mkdocs serve -a 127.0.0.1:8001