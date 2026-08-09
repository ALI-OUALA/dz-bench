"""Optional deterministic PNG rendering for the authored BAC-like corpus."""

from __future__ import annotations

import hashlib
import json
import random
from math import ceil
from pathlib import Path
from typing import Any

from .io import write_json
from .models import (
    AssetIndex,
    AssetRecord,
    Checksum,
    DatasetRevision,
    GroundTruth,
    GroundTruthDocument,
    Manifest,
    ManifestDocument,
    ManifestPage,
    PageContent,
)
from .synthetic import HEIGHT, WIDTH, SyntheticCorpus, generate_bac_corpus


def _load_optional_dependencies() -> tuple[Any, ...]:
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "PNG rendering needs the optional 'raster' extra; run `uv sync --extra raster`."
        ) from exc
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise RuntimeError(
            "Arabic PNG rendering needs arabic-reshaper and python-bidi; "
            "run `uv sync --extra raster`."
        ) from exc
    return Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, arabic_reshaper, get_display


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in text)


def _display_text(text: str, reshaper: Any, get_display: Any) -> str:
    if not _has_arabic(text):
        return text
    return get_display(reshaper.reshape(text))


def _wrap_text(draw: Any, text: str, font: Any, width: float) -> str:
    wrapped: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if not line or draw.textlength(candidate, font=font) <= width:
                line = candidate
            else:
                wrapped.append(line)
                line = word
        wrapped.append(line)
    return "\n".join(wrapped)


def _font_for_box(
    image_font: Any,
    font_path: Path,
    draw: Any,
    text: str,
    width: int,
    height: int,
    size_hint: int,
) -> tuple[Any, str]:
    for size in range(max(12, size_hint), 9, -2):
        font = image_font.truetype(str(font_path), size)
        wrapped = _wrap_text(draw, text, font, max(width - 24, 20))
        text_box = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4)
        if text_box[2] - text_box[0] <= width - 16 and text_box[3] - text_box[1] <= height - 16:
            return font, wrapped
    font = image_font.truetype(str(font_path), 10)
    return font, _wrap_text(draw, text, font, max(width - 16, 20))


def _draw_text_box(
    draw: Any,
    image_font: Any,
    reshaper: Any,
    get_display: Any,
    font_path: Path,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    direction: str,
    size_hint: int,
) -> None:
    displayed = _display_text(text, reshaper, get_display)
    font, wrapped = _font_for_box(image_font, font_path, draw, displayed, width, height, size_hint)
    anchor = "ra" if direction == "rtl" else "la"
    text_x = x + width - 10 if direction == "rtl" else x + 10
    draw.multiline_text(
        (text_x, y + 8),
        wrapped,
        font=font,
        fill=(20, 24, 30),
        spacing=4,
        anchor=anchor,
    )


def _block_direction(block: Any) -> str:
    return block.lines[0].direction if block.lines else "ltr"


def _block_text(block: Any) -> str:
    if block.equation_text:
        return block.equation_text
    return "\n".join(line.raw_text for line in block.lines)


def _draw_table(
    layer: Any,
    block: Any,
    image_draw: Any,
    image_font: Any,
    reshaper: Any,
    get_display: Any,
    font_path: Path,
) -> None:
    draw = image_draw.Draw(layer)
    block_x = block.bbox.x
    block_y = block.bbox.y
    block_width = ceil(block.bbox.width)
    block_height = ceil(block.bbox.height)
    draw.rectangle((0, 0, block_width - 1, block_height - 1), outline=(40, 45, 52), width=2)
    for cell in block.table.cells:
        x = round(cell.bbox.x - block_x)
        y = round(cell.bbox.y - block_y)
        width = max(1, round(cell.bbox.width))
        height = max(1, round(cell.bbox.height))
        draw.rectangle((x, y, x + width, y + height), outline=(75, 80, 88), width=1)
        _draw_text_box(
            draw,
            image_font,
            reshaper,
            get_display,
            font_path,
            cell.normalized_text,
            x,
            y,
            width,
            height,
            _block_direction(block),
            22,
        )


def _draw_diagram(layer: Any, image_draw: Any, category: str) -> None:
    draw = image_draw.Draw(layer)
    width, height = layer.size
    margin = max(20, min(width, height) // 12)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(45, 55, 70), width=3)
    if category == "physics":
        draw.line(
            (margin * 2, height - margin * 2, width - margin, height - margin * 2),
            fill=(25, 35, 50),
            width=3,
        )
        draw.line(
            (margin * 2, height - margin * 2, margin * 2, margin * 2), fill=(25, 35, 50), width=3
        )
        points = [
            (margin * 2, height - margin * 2),
            (width // 2, height // 2),
            (width - margin * 2, margin * 2),
        ]
        draw.line(points, fill=(165, 45, 45), width=5)
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(165, 45, 45))
    else:
        points = [
            (width // 2, margin * 2),
            (width - margin * 2, height - margin * 2),
            (margin * 2, height - margin * 2),
        ]
        draw.polygon(points, outline=(45, 75, 130), width=4)
        draw.line(
            (width // 2, margin * 2, width // 2, height - margin * 2), fill=(45, 75, 130), width=3
        )
        draw.ellipse(
            (width // 2 - 8, height // 2 - 8, width // 2 + 8, height // 2 + 8), fill=(45, 75, 130)
        )


def _render_block(
    page: PageContent,
    block: Any,
    font_path: Path,
    quality: str,
    modules: tuple[Any, ...],
) -> Any:
    image, image_draw, _, _, image_font, reshaper, get_display = modules
    width = max(1, ceil(block.bbox.width))
    height = max(1, ceil(block.bbox.height))
    layer = image.new("RGBA", (width, height), (255, 255, 255, 0))
    category = str(page.provenance.details.get("category", ""))
    if block.block_type == "table" and block.table is not None:
        _draw_table(layer, block, image_draw, image_font, reshaper, get_display, font_path)
    elif block.block_type == "diagram":
        _draw_diagram(layer, image_draw, category)
    elif block.lines or block.equation_text:
        draw = image_draw.Draw(layer)
        _draw_text_box(
            draw,
            image_font,
            reshaper,
            get_display,
            font_path,
            _block_text(block),
            0,
            0,
            width,
            height,
            _block_direction(block),
            38 if block.block_type in {"title", "equation"} else 28,
        )
        if block.block_type == "equation":
            draw.line((16, height - 12, width - 16, height - 12), fill=(60, 70, 80), width=2)
    if quality == "skewed":
        layer = layer.transform(
            layer.size,
            image.Transform.AFFINE,
            (1, -0.035, 0, 0, 1, 0),
            resample=image.Resampling.BICUBIC,
        )
    return layer


def _render_page(
    page: PageContent, font_path: Path, quality: str, seed: int, modules: tuple[Any, ...]
) -> Any:
    image, _, image_enhance, image_filter, _, _, _ = modules
    rendered = image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    for block in sorted(page.blocks, key=lambda item: item.reading_order_index):
        layer = _render_block(page, block, font_path, quality, modules)
        rendered.paste(layer, (round(block.bbox.x), round(block.bbox.y)), layer)
    if quality == "compressed":
        rendered = rendered.resize((WIDTH // 2, HEIGHT // 2), image.Resampling.LANCZOS)
        rendered = rendered.resize((WIDTH, HEIGHT), image.Resampling.LANCZOS)
        rendered = rendered.filter(image_filter.GaussianBlur(0.35))
        rendered = image_enhance.Contrast(rendered).enhance(0.9)
    elif quality == "photographed":
        paper = image.new("RGB", (WIDTH, HEIGHT), (241, 238, 229))
        rendered = image.blend(rendered, paper, 0.08)
        noise_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{page.page_id}".encode()).digest()[:8], "big"
        )
        noise = image.frombytes(
            "L", (WIDTH, HEIGHT), random.Random(noise_seed).randbytes(WIDTH * HEIGHT)
        ).convert("RGB")
        rendered = image.blend(rendered, noise, 0.018)
        rendered = rendered.filter(image_filter.GaussianBlur(0.25))
    elif quality == "skewed":
        pass
    elif quality != "clean":
        raise ValueError(f"unsupported synthetic scan quality: {quality}")
    return rendered


def _json_checksum(value: object) -> Checksum:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return Checksum(value=hashlib.sha256(payload).hexdigest())


def _png_checksum(path: Path) -> Checksum:
    data = path.read_bytes()
    return Checksum(value=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def _raster_manifest_and_truth(
    corpus: SyntheticCorpus,
    page_checksums: dict[str, Checksum],
    relative_paths: dict[str, str],
) -> tuple[Manifest, GroundTruth]:
    revision = DatasetRevision(dataset_id="bac-synthetic-images", revision="0.3.0")
    documents: list[ManifestDocument] = []
    truth_documents: list[GroundTruthDocument] = []
    truth_by_document = {
        document.document_id: document for document in corpus.ground_truth.documents
    }
    for document in corpus.manifest.documents:
        pages: list[ManifestPage] = []
        truth_pages: list[PageContent] = []
        for manifest_page in document.pages:
            checksum = page_checksums[manifest_page.page_id]
            pages.append(
                manifest_page.model_copy(
                    update={
                        "checksum": checksum,
                        "source_kind": "image",
                        "tags": [*manifest_page.tags, "raster-png"],
                    }
                )
            )
            truth_page = next(
                page
                for page in truth_by_document[document.document_id].pages
                if page.page_id == manifest_page.page_id
            ).model_copy(deep=True)
            truth_page.checksum = checksum
            truth_provenance = truth_page.provenance.model_copy(deep=True)
            truth_provenance.details.update(
                {"asset_format": "png", "image_path": relative_paths[manifest_page.page_id]}
            )
            truth_page.provenance = truth_provenance
            truth_pages.append(truth_page)
        document_checksum = _json_checksum(
            {"document_id": document.document_id, "pages": [page.checksum.value for page in pages]}
        )
        documents.append(
            document.model_copy(update={"checksum": document_checksum, "pages": pages})
        )
        truth_documents.append(
            GroundTruthDocument(document_id=document.document_id, pages=truth_pages)
        )
    manifest = corpus.manifest.model_copy(
        update={
            "dataset_revision": revision,
            "documents": documents,
            "notes": [
                *corpus.manifest.notes,
                "PNG bytes are the page checksums for this raster revision.",
                "Asset paths in assets.json are relative to the bundle root.",
            ],
        }
    )
    ground_truth = corpus.ground_truth.model_copy(
        update={"dataset_revision": revision, "documents": truth_documents}
    )
    return manifest, ground_truth


def write_bac_images(
    output_dir: str | Path,
    font_path: str | Path,
    seed: int = 17,
    repeats: int = 1,
) -> dict[str, Path]:
    """Render original BAC-like records to PNG and return a public bundle."""

    font = Path(font_path)
    if not font.is_file():
        raise FileNotFoundError(f"font file does not exist: {font}")
    modules = _load_optional_dependencies()
    corpus = generate_bac_corpus(seed=seed, repeats=repeats)
    directory = Path(output_dir)
    image_directory = directory / "images"
    image_directory.mkdir(parents=True, exist_ok=True)
    page_checksums: dict[str, Checksum] = {}
    relative_paths: dict[str, str] = {}
    records: list[dict[str, object]] = []
    assets: list[AssetRecord] = []
    records_by_page = {record["page_id"]: record for record in corpus.records}
    for document in corpus.ground_truth.documents:
        for page in document.pages:
            record = records_by_page[page.page_id]
            quality = str(record["scan_quality"])
            image_path = image_directory / f"{page.page_id}.png"
            rendered = _render_page(page, font, quality, seed, modules)
            rendered.save(image_path, format="PNG", optimize=False, compress_level=9)
            checksum = _png_checksum(image_path)
            relative_path = image_path.relative_to(directory).as_posix()
            page_checksums[page.page_id] = checksum
            relative_paths[page.page_id] = relative_path
            updated_record = dict(record)
            updated_record.update(
                {
                    "asset_format": "png",
                    "image_path": relative_path,
                    "image_sha256": checksum.value,
                    "image_size_bytes": checksum.size_bytes,
                    "width": WIDTH,
                    "height": HEIGHT,
                    "font_filename": font.name,
                    "font_is_bundled": False,
                }
            )
            records.append(updated_record)
            assets.append(
                AssetRecord(
                    document_id=document.document_id,
                    page_id=page.page_id,
                    media_type="image/png",
                    relative_path=relative_path,
                    checksum=checksum,
                    width=WIDTH,
                    height=HEIGHT,
                )
            )
    manifest, ground_truth = _raster_manifest_and_truth(corpus, page_checksums, relative_paths)
    asset_index = AssetIndex(dataset_revision=manifest.dataset_revision, assets=assets)
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "manifest": write_json(manifest, directory / "manifest.json"),
        "ground_truth": write_json(ground_truth, directory / "ground-truth.json"),
        "records": write_json(records, directory / "records.json"),
        "assets": write_json(asset_index, directory / "assets.json"),
        "images": image_directory,
    }


__all__ = ["write_bac_images"]
