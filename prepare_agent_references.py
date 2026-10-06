"""Render expert slides without losing arrows, crops, or embedded EMF images."""
import argparse
import hashlib
import json
import posixpath
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

SLIDES = {
    8: {"age": 0, "meaning": "No completed first annulus; 0+."},
    9: {"age": 1, "meaning": "One completed annulus; 1+."},
    10: {"age": 2, "meaning": "Two completed annuli; 2+."},
    11: {"age": 3, "meaning": "Three completed annuli; 3+."},
    12: {"age": 3, "meaning": "Four completed annuli; 4+, grouped into 3 or older."},
    13: {"age": None, "meaning": "Not Ring and Broken examples. Not Ring alone is NOT a bad label."},
}


def prepare(pptx, out, pdf=None):
    pptx, out = Path(pptx).resolve(), Path(out).resolve()
    if (out / "manifest.json").exists():
        raise FileExistsError("Reference package already exists; choose another output directory.")
    out.mkdir(parents=True, exist_ok=True)
    if not shutil.which("pdftoppm") or not shutil.which("pdfinfo"):
        raise RuntimeError("Install poppler (pdftoppm and pdfinfo) to render references.")
    with zipfile.ZipFile(pptx) as archive:
        presentation = ET.fromstring(archive.read("ppt/presentation.xml"))
        relationships = ET.fromstring(archive.read("ppt/_rels/presentation.xml.rels"))
        mapping = {r.attrib["Id"]: posixpath.normpath("ppt/" + r.attrib["Target"]) for r in relationships}
        slide_paths = [mapping[s.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
                       for s in presentation.findall(".//{http://schemas.openxmlformats.org/presentationml/2006/main}sldId")]
    with tempfile.TemporaryDirectory(prefix="trout-slides-") as temp:
        if pdf is None:
            office = shutil.which("soffice") or shutil.which("libreoffice")
            if not office:
                raise RuntimeError("Provide --pdf exported from PowerPoint, or install LibreOffice.")
            profile = Path(temp, "profile").as_uri()
            subprocess.run([office, f"-env:UserInstallation={profile}", "--headless",
                            "--convert-to", 'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"}}',
                            "--outdir", temp, str(pptx)], check=True)
            pdf = Path(temp, pptx.stem + ".pdf")
        pdf = Path(pdf).resolve()
        if not pdf.is_file():
            raise FileNotFoundError(pdf)
        details = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True, text=True).stdout
        page_count = next(int(line.split(":", 1)[1]) for line in details.splitlines() if line.startswith("Pages:"))
        if page_count != len(slide_paths):
            raise ValueError("PDF must include ALL slides including hidden slides; otherwise page numbers shift.")
        rows = []
        with zipfile.ZipFile(pptx) as archive:
            for slide, info in SLIDES.items():
                root = ET.fromstring(archive.read(slide_paths[slide-1]))
                text = "\n".join(t.text or "" for t in root.findall(
                    ".//{http://schemas.openxmlformats.org/drawingml/2006/main}t"))
                destination = out / f"slide_{slide:02d}"
                subprocess.run(["pdftoppm", "-f", str(slide), "-l", str(slide),
                                "-singlefile", "-scale-to", "1600", "-png",
                                str(pdf), str(destination)], check=True)
                image = destination.with_suffix(".png")
                rows.append({"slide": slide, "image": image.name, "text": text, **info,
                             "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
        manifest = {"source": pptx.name, "source_sha256": hashlib.sha256(pptx.read_bytes()).hexdigest(),
                    "slides": rows, "reference_fish_overlap": "unknown; audit before confirmatory evaluation",
                    "note": "Rendered examples are expert references, not model fine-tuning."}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Prepared:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pptx", required=True)
    parser.add_argument("--out", default="agent_references")
    parser.add_argument("--pdf", help="Optional matching PDF exported from PowerPoint.")
    args = parser.parse_args()
    prepare(args.pptx, args.out, args.pdf)
