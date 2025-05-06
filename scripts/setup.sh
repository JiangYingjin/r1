conda create -n r1 python=3.12 -y
conda activate r1
pip install uv
uv pip install -r requirements.txt
