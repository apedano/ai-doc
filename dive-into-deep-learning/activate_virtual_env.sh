python3 -m venv .venv
source .venv/bin/activate
#Extracted with ip freeze > requirements.txt
pip install -r ../requirements.txt
pip install mkdocs-material[imaging]
# shellcheck disable=SC2034
export NO_MKDOCS_2_WARNING=1
echo "Starting mkdocs serve"
echo "Don't forget to deploy GitHub"
echo "mkdocs serve -a 127.0.0.1:8001"
echo "Currently deployed page: https://apedano.github.io/python-course/"
# shellcheck disable=SC2164
#cd ./dive-into-deep-learning
echo "Killing the process running already on port 8001"
pids=$(lsof -t -i :8001 2>/dev/null)
if [ -z "$pids" ]; then
  echo "No process running on port 8001"
else
  echo "Killing process(es): $pids"
  kill -9 $pids
fi
echo "Starting mkdocs"
mkdocs serve -a 127.0.0.1:8001