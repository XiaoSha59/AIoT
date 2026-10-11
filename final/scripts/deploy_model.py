from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from raspberry_pi.inference import Classifier

FILES = ["face_classifier.onnx", "classes.json", "preprocessing.json", "threshold.json", "export_validation.json"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a verified bundle, smoke-test on Pi, then atomically switch active symlink")
    parser.add_argument("--host", default="pi@raspberrypi.local")
    parser.add_argument("--directory", default="/home/pi/edgeface")
    parser.add_argument("--model", type=Path, default=Path("models/active"))
    args = parser.parse_args()
    if not re.fullmatch(r"/home/[\w-]+/[\w/-]+", args.directory):
        raise ValueError("Unsupported remote path")
    classifier = Classifier(args.model)
    report = json.loads((args.model / "export_validation.json").read_text())
    digest = hashlib.sha256((args.model / "face_classifier.onnx").read_bytes()).hexdigest()
    if digest != report["onnx_sha256"] or report["checker"] != "passed":
        raise ValueError("Model export integrity failed")
    version = classifier.bundle_version
    release = f"{args.directory}/models/releases/{version}"
    subprocess.run(["ssh", args.host, f"mkdir -p '{release}'"], check=True)
    subprocess.run(["scp", *[str(args.model / file) for file in FILES], f"{args.host}:{release}/"], check=True)
    code = ("from pathlib import Path; import hashlib; from raspberry_pi.inference import Classifier; "
            f"p=Path('{release}'); assert hashlib.sha256((p/'face_classifier.onnx').read_bytes()).hexdigest()=='{digest}'; "
            "m=Classifier(p); print('Pi ONNX bundle smoke test passed:',m.version)")
    import shlex
    subprocess.run(["ssh", args.host, f"cd '{args.directory}' && .venv/bin/python -c {shlex.quote(code)}"], check=True)
    # Refuse replacing an existing directory; releases remain available for rollback.
    command = (f"cd '{args.directory}/models' && "
               f"if [ -d active ] && [ ! -L active ]; then echo 'active is a directory; archive it first' >&2; exit 1; fi; "
               f"ln -sfn 'releases/{version}' active.next && mv -Tf active.next active")
    subprocess.run(["ssh", args.host, command], check=True)
    print(f"Model activated: {version}. Send /stop then /start to load it.")


if __name__ == "__main__":
    main()
