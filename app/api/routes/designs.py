from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from typing import List, Optional, Any
import torch
import io
import json
from pathlib import Path
import re
from app.models import Design
from app.core import db as app_db

# Try to import torch_geometric for proper graph handling
try:
    from torch_geometric.data import Data as TGData
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

router = APIRouter(prefix="/designs", tags=["designs"])

# Metadata search paths - adjust as needed for your project structure
METADATA_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent.parent / "hierarchical-marl-chip-placement" / "src" / "data" / "preprocessed",
    Path(__file__).parent.parent.parent.parent / "hierarchical-marl-chip-placement" / "outputs",
]


def find_metadata(design_name: str) -> Optional[dict]:
    """Find metadata for a design by searching common paths."""
    for search_path in METADATA_SEARCH_PATHS:
        if not search_path.exists():
            continue
        
        # Search recursively for metadata files matching the design name
        for metadata_file in search_path.rglob(f"metadata-{design_name}-*.json"):
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except Exception:
                continue
        
        for metadata_file in search_path.rglob(f"metadata-*{design_name}*.json"):
            try:
                with open(metadata_file) as f:
                    return json.load(f)
            except Exception:
                continue
    
    return None


# ── Macro-kind inference helpers ──────────────────────────────────────────────

# Map LEF cell-name patterns to frontend MacroKind values.
_LEF_KIND_RULES: list[tuple[str, str]] = [
    ("fakeram",  "sram"),
    ("sram",     "sram"),
    ("ram",      "sram"),
    ("rom",      "sram"),
    ("regfile",  "reg"),
    ("rf_",      "reg"),
    ("reg",      "reg"),
    ("pll",      "pll"),
    ("clk",      "pll"),
    ("dsp",      "dsp"),
    ("mac",      "dsp"),      # multiply-accumulate
    ("npu",      "accel"),
    ("accel",    "accel"),
    ("noc",      "noc"),
    ("router",   "noc"),
    ("io",       "io"),
    ("pad",      "io"),
    ("buf",      "io"),
    ("cache",    "cache"),
    ("gpu",      "gpu"),
    ("core",     "cpu"),
    ("cpu",      "cpu"),
]

_VALID_KINDS = {"cpu", "sram", "cache", "dsp", "gpu", "accel", "pll", "io", "noc", "reg"}


def _lef_name_to_kind(lef_name: str) -> str:
    """Derive a frontend MacroKind from a LEF cell name."""
    lower = lef_name.lower()
    for pattern, kind in _LEF_KIND_RULES:
        if pattern in lower:
            return kind
    return "cpu"


def _build_dim_to_cell_lut(parsed_lef_macros: dict) -> dict[tuple[float, float], str]:
    """Build a lookup (width_um, height_um) → LEF cell name from metadata."""
    lut: dict[tuple[float, float], str] = {}
    for cell_name, dims in parsed_lef_macros.items():
        w = round(float(dims.get("width", 0)), 2)
        h = round(float(dims.get("height", 0)), 2)
        if w > 0 and h > 0:
            lut[(w, h)] = cell_name
    return lut


def _infer_kind_from_features(norm_w: float, norm_h: float) -> str:
    """Heuristic fallback: guess kind from normalised w/h when no metadata."""
    area = norm_w * norm_h
    aspect = norm_w / norm_h if norm_h > 0 else 1.0
    if area > 0.04:
        return "cpu"       # very large block
    if area > 0.015:
        return "accel"     # large block
    if aspect > 2.0 or aspect < 0.5:
        return "sram"      # tall/wide = memory
    if area < 0.002:
        return "io" if aspect > 0.8 else "pll"
    return "sram"          # default for chip-placement graphs


# ── Core conversion ──────────────────────────────────────────────────────────

def convert_torch_geometric_to_design(
    data: Any,
    filename: str,
    metadata: Optional[dict] = None,
) -> dict:
    """Convert a torch_geometric Data object to design format.
    
    Expected graph structure (ariane133 / Nangate45):
      data.pos  : [N, 2]  – already-normalized (0..1) macro positions
      data.x    : [N, 5]  – [width_um, height_um, area_um2, norm_w, norm_h]
      data.edge_index : [2, E]
    
    When *metadata* is supplied and contains ``parsed_lef_macros``, each node's
    raw µm dimensions (x[0], x[1]) are matched against the LEF cells to derive
    a meaningful ``kind`` (sram, cache, reg …).  Otherwise a heuristic based on
    area / aspect ratio is used.
    """
    if not HAS_TORCH_GEOMETRIC:
        return {
            "name": filename.replace(".pt", ""),
            "macros": [],
            "edges": [],
            "placements": [],
        }
    
    # ── Dimension → cell name lookup from metadata ────────────────────────
    dim_lut: dict[tuple[float, float], str] = {}
    global_kind: Optional[str] = None
    global_cell_types: list[str] = []

    if metadata:
        lef_macros = metadata.get("parsed_lef_macros", {})
        if lef_macros and isinstance(lef_macros, dict):
            dim_lut = _build_dim_to_cell_lut(lef_macros)

        # Parse known_info.macro_type to find true LEF types
        known = metadata.get("known_info", {})
        macro_type_str = (known.get("macro_type") or "").lower() if isinstance(known, dict) else ""

        if "sram" in macro_type_str:
            global_kind = "sram"
        elif "reg" in macro_type_str:
            global_kind = "reg"
        elif "cache" in macro_type_str:
            global_kind = "cache"
            
        # Extract shapes like "256x16" from "uniform 256x16-bit SRAM"
        matches = re.findall(r'(\d+x\d+)', macro_type_str)
        if matches and isinstance(lef_macros, dict):
            for m in matches:
                for k in lef_macros.keys():
                    if m in k:
                        if k not in global_cell_types:
                            global_cell_types.append(k)

    # ── Extract tensors ─────────────────────────────────────────────────
    has_pos = hasattr(data, "pos") and data.pos is not None
    has_x   = hasattr(data, "x")   and data.x   is not None

    pos_data = data.pos.numpy() if has_pos and hasattr(data.pos, "numpy") else (data.pos if has_pos else None)
    x_data   = data.x.numpy()   if has_x   and hasattr(data.x,   "numpy") else (data.x   if has_x   else None)

    num_nodes = len(pos_data) if pos_data is not None else (len(x_data) if x_data is not None else 0)

    macros = []
    for i in range(num_nodes):
        # Position: from pos tensor if available, else fallback to x[0]/x[1]
        if pos_data is not None:
            px = float(pos_data[i][0])
            py = float(pos_data[i][1])
        elif x_data is not None and len(x_data[i]) >= 2:
            px = float(x_data[i][0])
            py = float(x_data[i][1])
        else:
            px, py = 0.0, 0.0

        # Size: x[3] and x[4] are normalized w/h; fall back to tiny default
        if x_data is not None and len(x_data[i]) >= 5:
            pw = float(x_data[i][3])  # norm_w
            ph = float(x_data[i][4])  # norm_h
        elif x_data is not None and len(x_data[i]) >= 4:
            canvas = 400.0
            pw = float(x_data[i][0]) / canvas
            ph = float(x_data[i][1]) / canvas
        else:
            pw, ph = 0.02, 0.02

        # ── Determine kind & device_kind ────────────────────────────────
        device_kind = ""
        kind = "cpu"  # ultimate fallback
        
        # If there's an explicit type from the metadata macro_type string, use it
        # (Distribute cyclically if there are multiple types)
        if global_cell_types:
            device_kind = global_cell_types[i % len(global_cell_types)]
            kind = _lef_name_to_kind(device_kind)
        elif x_data is not None and len(x_data[i]) >= 2 and dim_lut:
            # Fallback to dimensions (often uninformative in this dataset)
            w_um = round(float(x_data[i][0]), 2)
            h_um = round(float(x_data[i][1]), 2)
            device_kind = dim_lut.get((w_um, h_um), "")
            if device_kind:
                kind = _lef_name_to_kind(device_kind)
                
        if not kind or kind == "cpu":
            if global_kind:
                kind = global_kind
            else:
                kind = _infer_kind_from_features(pw, ph)
                
        if kind not in _VALID_KINDS:
            kind = "cpu"

        # Clamp to valid range
        px = max(0.0, min(1.0, px))
        py = max(0.0, min(1.0, py))
        pw = max(0.001, min(1.0, pw))
        ph = max(0.001, min(1.0, ph))

        macros.append({
            "id": f"node_{i}",
            "name": f"node_{i}",
            "kind": kind,
            "deviceType": f"fd_{device_kind}",
            "x": round(px, 5),
            "y": round(py, 5),
            "w": round(pw, 5),
            "h": round(ph, 5),
            "area": round(pw * ph, 6),
        })
    
    # Extract edges
    edges = []
    if hasattr(data, "edge_index") and data.edge_index is not None:
        edge_index = data.edge_index.numpy() if hasattr(data.edge_index, "numpy") else data.edge_index
        for source, target in edge_index.T:
            edges.append({
                "source": f"node_{int(source)}",
                "target": f"node_{int(target)}",
            })
    
    return {
        "name": filename.replace(".pt", "").replace(".graph", ""),
        "macros": macros,
        "edges": edges,
        "placements": [
            {
                "label": "initial",
                "macros": macros,
            }
        ] if macros else [],
    }


def extract_design_from_pt(pt_data: bytes, filename: str) -> dict:
    """Extract design data from a PyTorch .pt file."""
    try:
        buffer = io.BytesIO(pt_data)
        loaded = torch.load(buffer, map_location="cpu", weights_only=False)
        
        # Handle torch_geometric Data objects directly
        if HAS_TORCH_GEOMETRIC and isinstance(loaded, TGData):
            return convert_torch_geometric_to_design(loaded, filename)
        
        # Handle dict — check for nested 'graph' key first (ariane133 / real-connection format)
        if isinstance(loaded, dict):
            if "graph" in loaded:
                graph_data = loaded["graph"]
                # Extract metadata from the dict if present
                pt_metadata = loaded.get("metadata") if isinstance(loaded.get("metadata"), dict) else None
                if HAS_TORCH_GEOMETRIC and isinstance(graph_data, TGData):
                    result = convert_torch_geometric_to_design(graph_data, filename, metadata=pt_metadata)
                    # Also attach metadata to the result for the frontend
                    if pt_metadata:
                        result["metadata"] = pt_metadata
                    print(pt_metadata)
                    return result
                elif isinstance(graph_data, dict):
                    return {
                        **graph_data,
                        "name": graph_data.get("name", filename.replace(".pt", "")),
                    }

            # Plain dict — check if it looks like design data
            design_keys = {"macros", "nodes", "placements", "edges"}
            if any(key in loaded for key in design_keys):
                return loaded
            if "data" in loaded and isinstance(loaded["data"], dict):
                return loaded["data"]
            if "graphData" in loaded:
                return loaded["graphData"]

        # Fallback: convert arbitrary object to dict
        if hasattr(loaded, "to_dict"):
            loaded = loaded.to_dict()
        elif not isinstance(loaded, dict):
            loaded = dict(loaded) if hasattr(loaded, "__iter__") else {"_raw": str(loaded)}

        return {
            "name": filename.replace(".pt", "").replace(".graph", ""),
            "macros": [],
            "edges": [],
            "placements": [],
        }
    except Exception as e:
        raise ValueError(f"Failed to deserialize .pt file: {str(e)}")




@router.get("/", response_model=List[dict])
async def list_all_designs():
    return await app_db.list_designs()


@router.post("/", status_code=201)
async def create_design(design: Design):
    saved = await app_db.create_design(design)
    if not saved or "Exception" in saved:
        raise HTTPException(status_code=500, detail="Failed to save design")
    return {"id": saved.get("id")}


@router.post("/upload-file")
async def upload_design_file(
    file: UploadFile = File(...),
    metadata_file: Optional[UploadFile] = File(None),
    name: Optional[str] = None,
):
    """Upload a design file (.pt, .json, or .graph) and optional metadata.
    
    Args:
        file: The main design file (.pt, .json, or .graph)
        metadata_file: Optional metadata JSON file
        name: Optional override for the design name
    
    Returns:
        Design data extracted from the file with attached metadata
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    filename = file.filename
    content = await file.read()
    
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    
    try:
        # Handle .pt files
        if filename.endswith(".pt") or filename.endswith(".graph.pt"):
            design_data = extract_design_from_pt(content, filename)
        # Handle JSON files
        elif filename.endswith(".json"):
            design_data = json.loads(content.decode("utf-8"))
        # Handle plain text graph files
        else:
            text_content = content.decode("utf-8")
            design_data = {
                "name": name or filename.replace("." + filename.split(".")[-1], ""),
                "content": text_content,
                "format": "text",
            }
        
        # Ensure name is set
        if "name" not in design_data or not design_data["name"]:
            design_data["name"] = name or filename.replace(".pt", "").replace(".json", "").replace(".graph", "")
        
        # Try to attach metadata from uploaded file first
        if metadata_file and metadata_file.filename:
            try:
                metadata_content = await metadata_file.read()
                if metadata_content:
                    uploaded_metadata = json.loads(metadata_content.decode("utf-8"))
                    design_data["metadata"] = uploaded_metadata
            except Exception as e:
                print(f"Warning: Failed to parse uploaded metadata file: {str(e)}")
        
        # Fall back to searching for metadata if not provided
        if "metadata" not in design_data:
            design_name = design_data.get("name", "").lower()
            metadata = find_metadata(design_name)
            if metadata:
                design_data["metadata"] = metadata
        
        return design_data
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


@router.get("/metadata/{design_name}")
async def get_design_metadata(design_name: str):
    """Fetch metadata for a design by name.
    
    Args:
        design_name: The name of the design (e.g., "ariane133")
    
    Returns:
        Metadata dict if found
    """
    metadata = find_metadata(design_name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Metadata not found for {design_name}")
    return metadata


@router.get("/{design_id}")
async def get_design(design_id: str):
    doc = await app_db.get_design(design_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Design not found")
    return doc


@router.post("/{design_id}/results", status_code=201)
async def upload_result(design_id: str, result: dict):
    saved = await app_db.create_result(design_id, result)
    if not saved:
        raise HTTPException(status_code=500, detail="Failed to save result")
    return {"id": saved["id"]}


@router.get("/{design_id}/results")
async def get_results(design_id: str):
    rows = await app_db.list_results(design_id)
    return rows
