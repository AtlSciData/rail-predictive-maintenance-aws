"""Build build/sourcedir.tar.gz for a SageMaker training job (script mode).

Usage:  python sagemaker/package.py
Then upload build/sourcedir.tar.gz to s3://<bucket>/sagemaker/source/sourcedir.tar.gz
Contents (flat): train.py, requirements.txt, metrics.py (copied from src/rulpm/metrics.py)
"""
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
files = {
    "train.py": root / "sagemaker" / "train.py",
    "requirements.txt": root / "sagemaker" / "requirements.txt",
    "metrics.py": root / "src" / "rulpm" / "metrics.py",
}
out = root / "build"
out.mkdir(exist_ok=True)
with tarfile.open(out / "sourcedir.tar.gz", "w:gz") as tar:
    for name, path in files.items():
        tar.add(path, arcname=name)
print("wrote", out / "sourcedir.tar.gz", "->", ", ".join(files))
