python3 -m venv .venv
source .venv/bin/activate

pip install mkdocs-material[imaging]

echo "Starting mkdocs serve"
echo "Don't forget to deploy GitHub"
echo "mkdocs serve -a 127.0.0.1:8001"
echo "Currently deployed page: https://apedano.github.io/python-course/"
cd python-science
mkdocs serve -a 127.0.0.1:8001