"""
real_api.py — Production OMR API that runs the actual pipeline.

Wraps pipeline.run_pipeline() in a Flask service.  The response shapes are
identical to dummy_api.py so frontends can switch between them with only a
URL change.

Endpoints
---------
    POST /process                  — full pipeline: image → all outputs
    GET  /rectified                — last rectified PNG
    GET  /detections               — last detection JSON
    GET  /xml                      — last MusicXML
    GET  /full                     — last full response (all three combined)
    GET  /health                   — server + pipeline readiness check

    Per-step endpoints (run individual pipeline stages):
    POST /staff_detect/unet        — U-Net staff-line detection
    POST /staff_detect/classical   — algorithmic staff-line detection
    POST /note_align               — batch notehead/rest alignment refinement
    POST /note_align/click         — single-click refinement for editor UIs
    POST /xml_build                — build MusicXML from detections

The GET endpoints serve a cached "last result" so a frontend that already
called /process can re-fetch a single artifact without re-running the model.
For a stateless multi-user server, replace the LAST_RESULT global with a
session-keyed cache or require /process every time.

Configuration
-------------
    MODEL_PATH         path to YOLO weights
    OUTPUT_BASE_DIR    where pipeline writes intermediate files
    MAX_IMAGE_MB       reject uploads above this size
    HOST, PORT         listen address

All can be overridden via environment variables of the same name, e.g.:
    set MODEL_PATH=S:\omr\models\v2.pt
    python real_api.py

Usage
-----
    pip install flask
    python real_api.py

    # Submit an image:
    curl -F "image=@score.png" http://localhost:5000/process -o response.json
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import sys
import tempfile
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Bootstrap: register <project>/src and <project>/scripts on sys.path so
# this file (which lives under <project>/api/) can import env_loader and
# the entry-point pipeline module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_PROJECT_ROOT / 'scripts'))

import env_loader  # noqa: E402
import pipeline    # noqa: E402

# Per-step modules — imported lazily where needed (some are heavy: U-Net
# pulls in torch, build_musicxml_v2 pulls in lxml) but listed here for
# the IDE / readability.
#   src/staff_detector_unet.detect_staves_unet
#   src/staff_rectifier.detect_staff_curves     (classical detection mode)
#   src/note_alignment.{refine_detection, ALIGN_CONFIG, load_template}
#   src/build_musicxml_v2.build_score_xml_v2


# ─────────────────────────────────────────────────────────────────────────────
# JSON → ProcessedScore/PageDetections adapter
# ─────────────────────────────────────────────────────────────────────────────
#
# build_score_xml_v2 expects in-memory dataclasses, not JSON.  This
# adapter reconstructs the minimal-fidelity ProcessedScore and
# PageDetections shapes from a JSON request body, so an editor frontend
# can re-build MusicXML from edited detections without going through the
# full pipeline.
#
# Coordinate convention reminder: detections.json fields cx/cy/x1/y1/x2/y2
# are PAGE-relative.  The Detection dataclass expects them as CROP-relative
# (subtracted by the staff's crop_x1 / crop_y1), with separate full_cx/full_cy
# fields for the absolute coords.  This adapter handles the conversion.

def _build_pseudo_score(body: Dict[str, Any]):
    """
    Given a JSON request body of the shape::

        {
          "staves": [
            { "part_id": "P1", "staff_in_part": 0,
              "top_y": 363, "bot_y": 425, "left_x": 0, "right_x": 2549,
              "line_positions": [363, 378, 393, 410, 425],
              "line_spacing": 15.5,
              "crop_y1": 305, "crop_x1": 137,
              "crop_y2": 460, "crop_x2": 2549,    # optional
              "detections": [ {"class": ..., "cx": ..., "cy": ..., ...}, ... ]
            }, ...
          ],
          "num_parts": 1                          # optional, inferred from part_ids
        }

    Returns ``(ProcessedScore, PageDetections)`` with the minimal field set
    build_score_xml_v2 actually reads.  ``pstaff.cleaned`` / ``pstaff.crop``
    are 1x1 stub arrays — the XML builder doesn't touch pixel data.
    """
    import numpy as np
    from preprocessing import ProcessedScore, ProcessedStaff
    from score_analyzer import StaffData
    from symbol_detector import PageDetections, StaffDetections, Detection

    staves_json = body.get('staves') or []
    if not staves_json:
        raise ValueError("'staves' must be a non-empty list.")

    # Infer num_parts from part_ids if not explicit.
    declared = body.get('num_parts')
    if declared is None:
        part_ids = sorted({s.get('part_id', 'P1') for s in staves_json})
        num_parts = len(part_ids) or 1
    else:
        num_parts = int(declared)

    pstaves_by_part: List[List] = [[] for _ in range(num_parts)]
    sdets_by_part: List[List] = [[] for _ in range(num_parts)]
    all_pstaves: List = []
    all_sdets: List = []

    dummy_img = np.zeros((1, 1), dtype=np.uint8)

    img_w_max = 0
    img_h_max = 0

    for i, s in enumerate(staves_json):
        part_id = s.get('part_id', 'P1')
        try:
            part_idx = int(part_id.lstrip('P')) - 1   # 'P1' → 0
        except ValueError:
            part_idx = 0
        if part_idx < 0 or part_idx >= num_parts:
            # part_id outside declared num_parts — drop to last available
            part_idx = max(0, min(part_idx, num_parts - 1))

        crop_x1 = int(s.get('crop_x1', 0))
        crop_y1 = int(s.get('crop_y1', 0))
        crop_x2 = int(s.get('crop_x2', s.get('right_x', 0) or 0))
        crop_y2 = int(s.get('crop_y2', s.get('bot_y', 0) or 0))

        img_w_max = max(img_w_max, crop_x2, int(s.get('right_x', 0) or 0))
        img_h_max = max(img_h_max, crop_y2, int(s.get('bot_y', 0) or 0))

        staff_data = StaffData(
            staff_idx     = i,
            part_id       = part_id,
            staff_in_part = int(s.get('staff_in_part', 0)),
            top_y         = int(s.get('top_y', 0)),
            bot_y         = int(s.get('bot_y', 0)),
            left_x        = int(s.get('left_x', crop_x1)),
            right_x       = int(s.get('right_x', crop_x2)),
            line_spacing  = float(s.get('line_spacing', 0.0)),
            line_positions = list(s.get('line_positions') or []),
            crop_y1       = crop_y1,
            crop_y2       = crop_y2,
            crop_x1       = crop_x1,
            crop_x2       = crop_x2,
        )

        pstaff = ProcessedStaff(
            staff_data = staff_data,
            crop       = dummy_img,
            cleaned    = dummy_img,
        )
        pstaves_by_part[part_idx].append(pstaff)
        all_pstaves.append(pstaff)

        # Build Detection list (JSON page-rel → in-memory crop-rel + full)
        det_objs: List = []
        for d in s.get('detections') or []:
            full_cx = int(d['cx'])
            full_cy = int(d['cy'])
            full_x1 = int(d['x1'])
            full_y1 = int(d['y1'])
            full_x2 = int(d['x2'])
            full_y2 = int(d['y2'])
            det_objs.append(Detection(
                class_name = d.get('class') or d.get('class_name', ''),
                conf       = float(d.get('conf', 1.0)),
                cx = full_cx - crop_x1, cy = full_cy - crop_y1,
                x1 = full_x1 - crop_x1, y1 = full_y1 - crop_y1,
                x2 = full_x2 - crop_x1, y2 = full_y2 - crop_y1,
                full_cx = full_cx, full_cy = full_cy,
            ))

        sd_det = StaffDetections(
            part_id        = part_id,
            staff_in_part  = staff_data.staff_in_part,
            top_y          = staff_data.top_y,
            bot_y          = staff_data.bot_y,
            line_positions = staff_data.line_positions,
            line_spacing   = staff_data.line_spacing,
            crop_y1        = crop_y1,
            crop_x1        = crop_x1,
            detections     = det_objs,
        )
        sdets_by_part[part_idx].append(sd_det)
        all_sdets.append(sd_det)

    # Sort each part's lists top-to-bottom (the XML builder iterates in
    # order and assumes top-first within a part).
    for plist in pstaves_by_part:
        plist.sort(key=lambda p: p.staff_data.top_y)
    for slist in sdets_by_part:
        slist.sort(key=lambda s: s.top_y)
    all_pstaves.sort(key=lambda p: p.staff_data.top_y)
    all_sdets.sort(key=lambda s: s.top_y)

    processed_score = ProcessedScore(
        image_path = '',
        img_h      = img_h_max or 1,
        img_w      = img_w_max or 1,
        num_parts  = num_parts,
        parts      = pstaves_by_part,
        all_staves = all_pstaves,
    )
    page_detections = PageDetections(
        image_path = '',
        img_h      = img_h_max or 1,
        img_w      = img_w_max or 1,
        num_parts  = num_parts,
        parts      = sdets_by_part,
        all_staves = all_sdets,
    )
    return processed_score, page_detections


# ─────────────────────────────────────────────────────────────────────────────
# Configuration (env-var overridable)
# ─────────────────────────────────────────────────────────────────────────────

# Centralized path config lives in env_loader; pull it via the public
# constants so the API matches the rest of the pipeline.
MODEL_PATH      = Path(env_loader.MODEL_PATH)
OUTPUT_BASE_DIR = Path(env_loader.OUTPUT_BASE_DIR)
MAX_IMAGE_MB    = env_loader.MAX_IMAGE_MB
HOST            = env_loader.HOST
PORT            = env_loader.PORT

ALLOWED_EXT = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'}


# ─────────────────────────────────────────────────────────────────────────────
# Cached "last result" — for the GET endpoints
# ─────────────────────────────────────────────────────────────────────────────
#
# A single shared cache is fine for a single-user/local-tool deployment.
# Replace with a session/UUID-keyed dict if exposing this to multiple users.

_LAST_RESULT: Optional[Dict[str, Any]] = None
_PIPELINE_LOCK = threading.Lock()    # serialize model inference


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline call
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(image_bytes: bytes, original_name: str) -> Dict[str, Any]:
    """
    Save the upload to a fresh output directory, run the pipeline, and
    return the result paths plus metadata.  Caches the result globally.
    """
    global _LAST_RESULT

    # Per-request output directory keeps multiple submissions separated
    job_id  = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    job_dir = OUTPUT_BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save the upload using its original extension (or .png as fallback)
    ext = Path(original_name).suffix.lower() or '.png'
    if ext not in ALLOWED_EXT:
        ext = '.png'
    in_path = job_dir / f'input{ext}'
    in_path.write_bytes(image_bytes)

    # Inference is not thread-safe across concurrent calls — serialize it
    with _PIPELINE_LOCK:
        result = pipeline.run_pipeline(
            image_path        = str(in_path),
            output_dir        = str(job_dir),
            model_path        = str(MODEL_PATH),
            save_debug_images = False,
            save_labeled_crops = True,
        )

    # Augment with metadata (don't include the in-memory PageDetections
    # object — it isn't JSON-serializable and the JSON file already has it)
    response = {
        'job_id':           job_id,
        'rectified_image':  result['rectified_image'],
        'detections_json':  result['detections_json'],
        'xml_file':         result['xml_file'],
        'labeled_crops_dir': result.get('labeled_crops_dir'),
    }
    _LAST_RESULT = response
    return response


def _build_full_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read the pipeline's on-disk artifacts and pack them into the same
    JSON envelope as dummy_api.get_full_response().
    """
    rectified_path = Path(result['rectified_image'])
    xml_path       = Path(result['xml_file'])
    det_path       = Path(result['detections_json'])

    img_bytes = rectified_path.read_bytes()
    mime, _   = mimetypes.guess_type(str(rectified_path))

    with open(det_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    return {
        'job_id':               result['job_id'],
        'rectified_image_b64':  base64.b64encode(img_bytes).decode('ascii'),
        'rectified_image_mime': mime or 'image/png',
        'detections':           detections,
        'xml':                  xml_path.read_text(encoding='utf-8'),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────

def create_app():
    try:
        from flask import Flask, jsonify, request, send_file, Response
    except ImportError:
        raise ImportError(
            'Flask is required.  Install with:  pip install flask'
        )

    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_MB * 1024 * 1024

    @app.after_request
    def add_cors(response):
        response.headers['Access-Control-Allow-Origin']  = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response

    @app.route('/<path:p>', methods=['OPTIONS'])
    def options_handler(p):
        from flask import Response as R
        return R(status=204, headers={
            'Access-Control-Allow-Origin':  '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        })

    # ── GET / — serve the browser API tester ──
    @app.route('/', methods=['GET'])
    def tester_ui():
        tester_path = Path(__file__).parent / 'api_tester.html'
        if tester_path.exists():
            return send_file(str(tester_path), mimetype='text/html')
        return ('<p>api_tester.html not found next to real_api.py. '
                'Place both files in the same directory.</p>'), 404

    # ── POST /process ──
    @app.route('/process', methods=['POST'])
    def process_endpoint():
        """
        Submit an image, get the full response (rectified, detections, xml).

        Accepts:
            multipart/form-data with field name 'image'
            application/octet-stream  (raw image bytes; filename in URL ?name=)
        """
        try:
            # multipart/form-data
            if 'image' in request.files:
                f = request.files['image']
                image_bytes = f.read()
                original_name = f.filename or 'upload.png'
            # raw POST body
            elif request.data:
                image_bytes = request.data
                original_name = request.args.get('name', 'upload.png')
            else:
                return jsonify({
                    'error':   'no_image',
                    'message': "Send the image as multipart 'image' field "
                               "or raw body."
                }), 400

            if len(image_bytes) == 0:
                return jsonify({
                    'error':   'empty_upload',
                    'message': 'Image payload is empty.'
                }), 400

            result = _run_pipeline(image_bytes, original_name)
            return jsonify(_build_full_response(result))

        except Exception as e:
            traceback.print_exc()
            return jsonify({
                'error':   'pipeline_error',
                'message': str(e),
                'type':    type(e).__name__,
            }), 500

    # ── GET /rectified ──
    @app.route('/rectified', methods=['GET'])
    def rectified_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        return send_file(
            _LAST_RESULT['rectified_image'],
            mimetype='image/png',
            as_attachment=True,
            download_name='rectified.png',
        )

    # ── GET /detections ──
    @app.route('/detections', methods=['GET'])
    def detections_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        with open(_LAST_RESULT['detections_json'], 'r', encoding='utf-8') as f:
            return Response(f.read(), mimetype='application/json')

    # ── GET /xml ──
    @app.route('/xml', methods=['GET'])
    def xml_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        return send_file(
            _LAST_RESULT['xml_file'],
            mimetype='application/xml',
            as_attachment=True,
            download_name='score.xml',
        )

    # ── GET /full ──
    @app.route('/full', methods=['GET'])
    def full_endpoint():
        if _LAST_RESULT is None:
            return jsonify({
                'error':   'no_cached_result',
                'message': 'Call POST /process first.'
            }), 404
        return jsonify(_build_full_response(_LAST_RESULT))

    # ─────────────────────────────────────────────────────────────────────
    # Per-step endpoints — let callers run individual pipeline stages
    # without going through /process.
    #
    # Conventions
    # -----------
    # * Image-bearing endpoints accept the image as a multipart 'image'
    #   field OR raw request body, identical to /process.
    # * JSON-only endpoints accept application/json bodies.
    # * Mixed (image + JSON) endpoints take multipart with 'image' file
    #   field + 'params' string field containing the JSON payload.
    # * All errors return {"error": "<code>", "message": "..."} with an
    #   appropriate HTTP status.
    # * These endpoints DO NOT touch _LAST_RESULT and do not interfere
    #   with /process.  Each call is independent.
    # ─────────────────────────────────────────────────────────────────────

    def _read_image_arg(field_name: str = 'image'):
        """
        Extract raw image bytes from the current request, supporting
        either multipart-form upload or raw-body POST.  Returns
        (bytes, original_filename).  Raises ValueError if no image is
        present.
        """
        if field_name in request.files:
            f = request.files[field_name]
            return f.read(), (f.filename or 'upload.png')
        if request.data:
            return request.data, request.args.get('name', 'upload.png')
        raise ValueError(
            f"No image provided.  Send as multipart '{field_name}' field "
            f"or raw POST body.")

    def _decode_image_to_gray(image_bytes: bytes):
        """Bytes -> grayscale ndarray (uint8 HxW).  Raises ValueError."""
        import numpy as np
        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError('Could not decode image bytes.')
        return img

    # ── POST /staff_detect/unet ──
    # Run U-Net staff-line detection only.  No rectification, no
    # downstream processing.  Useful for callers who just want staff
    # coordinates for their own purposes (or to compare against
    # /staff_detect/classical).
    @app.route('/staff_detect/unet', methods=['POST'])
    def staff_detect_unet_endpoint():
        try:
            image_bytes, _ = _read_image_arg()
            if not image_bytes:
                return jsonify({'error': 'empty_upload',
                                'message': 'Image payload is empty.'}), 400
            img = _decode_image_to_gray(image_bytes)

            # Optional query params: threshold, inference_size
            threshold = float(request.args.get('threshold', 0.5))
            inference_size = int(request.args.get('inference_size', 512))

            from staff_detector_unet import (detect_staves_unet,
                                              is_unet_available)
            if not is_unet_available():
                return jsonify({
                    'error': 'unet_unavailable',
                    'message': ('U-Net checkpoint not found.  Configure '
                                'UNET_MODEL_PATH in .env.')
                }), 503

            with _PIPELINE_LOCK:
                staves = detect_staves_unet(
                    img,
                    threshold=threshold,
                    inference_size=inference_size,
                    verbose=False,
                )
            return jsonify({
                'method':   'unet',
                'image_w':  int(img.shape[1]),
                'image_h':  int(img.shape[0]),
                'n_staves': len(staves),
                'staves':   staves,
            })
        except ValueError as ve:
            return jsonify({'error': 'bad_request',
                            'message': str(ve)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': 'unet_error',
                            'message': str(e),
                            'type': type(e).__name__}), 500

    # ── POST /staff_detect/classical ──
    # Run the algorithmic (morphology + curve-tracing) staff detector
    # from staff_rectifier.detect_staff_curves with use_unet=False and
    # use_yolo=False so only the classical fallback runs.  Same response
    # schema as /staff_detect/unet — callers can A/B compare them.
    @app.route('/staff_detect/classical', methods=['POST'])
    def staff_detect_classical_endpoint():
        try:
            image_bytes, _ = _read_image_arg()
            if not image_bytes:
                return jsonify({'error': 'empty_upload',
                                'message': 'Image payload is empty.'}), 400
            img = _decode_image_to_gray(image_bytes)

            # Optional: force_strategy lets the caller pick a specific
            # classical preset ('printed', 'handwritten', 'uneven_scan',
            # 'permissive').  Default lets detect_staff_curves auto-pick.
            force_strategy = request.args.get('strategy')

            from staff_rectifier import detect_staff_curves
            staves = detect_staff_curves(
                img,
                verbose=False,
                use_unet=False,
                use_yolo=False,
                force_strategy=force_strategy,
            )
            return jsonify({
                'method':   'classical',
                'strategy': force_strategy or 'auto',
                'image_w':  int(img.shape[1]),
                'image_h':  int(img.shape[0]),
                'n_staves': len(staves),
                'staves':   staves,
            })
        except ValueError as ve:
            return jsonify({'error': 'bad_request',
                            'message': str(ve)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': 'classical_error',
                            'message': str(e),
                            'type': type(e).__name__}), 500

    # ── POST /note_align ──
    # Run template-based notehead / rest alignment on a list of
    # YOLO-style detections against the provided cleaned staff image.
    #
    # Multipart fields:
    #   image   the cleaned (staff-line-removed) crop the detections came from
    #   params  JSON string:
    #             {
    #               "detections": [
    #                 {"class": "noteheadBlack", "cx": 271, "cy": 102,
    #                  "x1": 257, "y1": 90, "x2": 286, "y2": 114},
    #                 ...
    #               ],
    #               "crop_x1": 0, "crop_y1": 0,    # optional, default 0
    #               "loss_threshold": null         # optional per-class override
    #             }
    #
    # Returns the same detection list, with cx/cy/x1/y1/x2/y2 updated
    # where alignment accepted the refinement, and a 'aligned' boolean
    # per detection.
    @app.route('/note_align', methods=['POST'])
    def note_align_endpoint():
        try:
            image_bytes, _ = _read_image_arg()
            if not image_bytes:
                return jsonify({'error': 'empty_upload',
                                'message': 'Image payload is empty.'}), 400
            img_gray = _decode_image_to_gray(image_bytes)

            # Parse params
            params_str = request.form.get('params') if request.form else None
            if not params_str:
                # also accept ?params=...&... query if caller prefers
                params_str = request.args.get('params')
            if not params_str:
                return jsonify({
                    'error': 'missing_params',
                    'message': ("Multipart 'params' field with JSON body "
                                "is required.")
                }), 400
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError as je:
                return jsonify({'error': 'bad_json',
                                'message': f'params is not valid JSON: {je}'}), 400

            detections = params.get('detections', [])
            if not isinstance(detections, list):
                return jsonify({'error': 'bad_request',
                                'message': "'detections' must be a list."}), 400
            crop_x1 = int(params.get('crop_x1', 0))
            crop_y1 = int(params.get('crop_y1', 0))
            loss_threshold = params.get('loss_threshold')   # None = per-class

            from note_alignment import (refine_detection,
                                         ALIGN_CONFIG,
                                         load_template)

            # Preload templates for the classes we'll touch
            templates = {}
            for cls in {d.get('class') for d in detections}:
                if cls in ALIGN_CONFIG:
                    tpl = load_template(cls)
                    if tpl is not None:
                        templates[cls] = tpl

            class _Det:
                __slots__ = ('class_name', 'cx', 'cy', 'x1', 'y1', 'x2', 'y2',
                             'full_cx', 'full_cy')
                def __init__(self, d):
                    self.class_name = d['class']
                    # JSON convention is page-relative; convert to crop-rel.
                    self.cx = int(d['cx']) - crop_x1
                    self.cy = int(d['cy']) - crop_y1
                    self.x1 = int(d['x1']) - crop_x1
                    self.x2 = int(d['x2']) - crop_x1
                    self.y1 = int(d['y1']) - crop_y1
                    self.y2 = int(d['y2']) - crop_y1
                    self.full_cx = self.cx + crop_x1
                    self.full_cy = self.cy + crop_y1

            refined_list = []
            n_modified = 0
            for d in detections:
                tpl = templates.get(d.get('class'))
                if tpl is None:
                    # Echo the detection unchanged when no template applies.
                    out = dict(d)
                    out['aligned'] = False
                    refined_list.append(out)
                    continue
                det = _Det(d)
                ok = refine_detection(
                    det, img_gray, tpl,
                    crop_y1=crop_y1, crop_x1=crop_x1,
                    loss_threshold=loss_threshold,
                )
                out = dict(d)
                out['cx'] = det.cx + crop_x1
                out['cy'] = det.cy + crop_y1
                out['x1'] = det.x1 + crop_x1
                out['y1'] = det.y1 + crop_y1
                out['x2'] = det.x2 + crop_x1
                out['y2'] = det.y2 + crop_y1
                out['aligned'] = bool(ok)
                if ok:
                    n_modified += 1
                refined_list.append(out)

            return jsonify({
                'n_input':      len(detections),
                'n_attempted':  sum(1 for d in detections
                                     if d.get('class') in templates),
                'n_modified':   n_modified,
                'crop_x1':      crop_x1,
                'crop_y1':      crop_y1,
                'templates_used': sorted(templates.keys()),
                'detections':   refined_list,
            })
        except ValueError as ve:
            return jsonify({'error': 'bad_request',
                            'message': str(ve)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': 'align_error',
                            'message': str(e),
                            'type': type(e).__name__}), 500

    # ── POST /note_align/click ──
    # Single-detection refinement around a click point.  Takes the
    # rectified page image, a click (x, y), and a YOLO class — crops a
    # local window around the click, binarizes it raw (no staffline
    # removal needed), and runs alignment with the existing templates.
    #
    # This is the editor's "click-to-snap" gesture.  The two-sided
    # Chamfer matching tolerates staffline noise in the local window
    # because the noise is roughly constant across candidate positions
    # in a small search radius.  See note_alignment.py for the algorithm.
    #
    # Multipart fields:
    #   image    rectified page (NOT cleaned — the cleaning step is
    #            unnecessary for small-window alignment)
    #   params   JSON string:
    #             {
    #               "x": 271, "y": 102,          # click position (page-rel)
    #               "class": "noteheadBlack",
    #               "init_w": 28, "init_h": 22,  # optional, default 30x24
    #               "window": 80,                # optional, crop side length
    #               "loss_threshold": null       # optional override
    #             }
    @app.route('/note_align/click', methods=['POST'])
    def note_align_click_endpoint():
        import numpy as np
        import cv2 as _cv2
        try:
            image_bytes, _ = _read_image_arg()
            if not image_bytes:
                return jsonify({'error': 'empty_upload',
                                'message': 'Image payload is empty.'}), 400
            img_full = _decode_image_to_gray(image_bytes)

            params_str = (request.form.get('params')
                          if request.form else None)
            if not params_str:
                params_str = request.args.get('params')
            if not params_str:
                return jsonify({
                    'error': 'missing_params',
                    'message': ("Multipart 'params' field with JSON body "
                                "is required.")
                }), 400
            try:
                params = json.loads(params_str)
            except json.JSONDecodeError as je:
                return jsonify({'error': 'bad_json',
                                'message': f'params is not valid JSON: {je}'}), 400

            try:
                click_x = int(params['x'])
                click_y = int(params['y'])
                cls = params['class']
            except (KeyError, TypeError, ValueError) as ke:
                return jsonify({'error': 'bad_request',
                                'message': f'params missing x/y/class: {ke}'}), 400

            init_w = int(params.get('init_w', 30))
            init_h = int(params.get('init_h', 24))
            window = int(params.get('window', 80))
            loss_threshold = params.get('loss_threshold')

            from note_alignment import (align_note_with_variants,
                                         ALIGN_CONFIG,
                                         load_template)
            cfg = ALIGN_CONFIG.get(cls)
            if cfg is None:
                return jsonify({
                    'error': 'unsupported_class',
                    'message': f"No ALIGN_CONFIG entry for class '{cls}'.",
                }), 400
            tpl = load_template(cls)
            if tpl is None:
                return jsonify({
                    'error': 'template_missing',
                    'message': (f"No template found for '{cls}' in "
                                 "data/note_templates/."),
                }), 503

            # Crop a local window around the click — bounds-checked.
            H, W = img_full.shape[:2]
            half = max(window, init_w + 20, init_h + 20) // 2
            x0 = max(0, click_x - half)
            y0 = max(0, click_y - half)
            x1 = min(W, click_x + half)
            y1 = min(H, click_y + half)
            local = img_full[y0:y1, x0:x1]
            if local.size == 0:
                return jsonify({
                    'error': 'bad_request',
                    'message': 'Click position is outside the image.'
                }), 400

            # The alignment expects an image with dark glyphs on light
            # paper.  For a rectified scan that's typically already the
            # case, but apply a light Otsu binarisation to normalise
            # contrast so the threshold parameter inside the alignment
            # behaves predictably.  Stafflines stay in — they're a
            # constant noise floor across the small window and don't
            # bias the search.
            _, local_bin = _cv2.threshold(
                local, 0, 255,
                _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)

            # Run the variant search around the click in LOCAL coords.
            local_click_x = click_x - x0
            local_click_y = click_y - y0
            result = align_note_with_variants(
                bg_gray            = local_bin,
                base_template_rgba = tpl,
                init_x             = local_click_x,
                init_y             = local_click_y,
                init_w             = init_w,
                init_h             = init_h,
                rotations          = cfg.get('rotations', [0]),
                scale_min          = cfg.get('scale_min', 0.85),
                scale_max          = cfg.get('scale_max', 1.15),
                scale_step         = cfg.get('scale_step', 0.05),
                max_shift          = cfg.get('max_shift', 8),
                threshold          = cfg.get('threshold', 150),
                inv_weight         = cfg.get('inv_weight', 1.0),
            )

            # Accept the refinement only when the loss is below the
            # per-class threshold (or the override).
            effective_threshold = (loss_threshold
                                   if loss_threshold is not None
                                   else cfg.get('loss_thresh', 1.0))
            accepted = result['loss'] <= float(effective_threshold)

            if accepted:
                # Convert back to page coords.
                refined_x = float(result['x']) + x0
                refined_y = float(result['y']) + y0
                mw = int(result.get('mask_w', init_w))
                mh = int(result.get('mask_h', init_h))
                # Box-of-final-mask in page coords.
                half_w = mw // 2
                half_h = mh // 2
                refined_box = {
                    'x1': int(round(refined_x - half_w)),
                    'y1': int(round(refined_y - half_h)),
                    'x2': int(round(refined_x + (mw - half_w))),
                    'y2': int(round(refined_y + (mh - half_h))),
                }
            else:
                refined_x = float(click_x)
                refined_y = float(click_y)
                refined_box = {
                    'x1': click_x - init_w // 2,
                    'y1': click_y - init_h // 2,
                    'x2': click_x + (init_w - init_w // 2),
                    'y2': click_y + (init_h - init_h // 2),
                }

            return jsonify({
                'class':            cls,
                'click':            {'x': click_x, 'y': click_y},
                'refined':          {'x': refined_x, 'y': refined_y},
                'box':              refined_box,
                'shift_px':         float(np.hypot(refined_x - click_x,
                                                    refined_y - click_y)),
                'loss':             float(result['loss']),
                'loss_threshold':   float(effective_threshold),
                'accepted':         bool(accepted),
                'scale':            float(result.get('scale', 1.0)),
                'rotation':         int(result.get('rotation', 0)),
                'window':           {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1},
            })
        except ValueError as ve:
            return jsonify({'error': 'bad_request',
                            'message': str(ve)}), 400
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': 'click_error',
                            'message': str(e),
                            'type': type(e).__name__}), 500

    # ── POST /xml_build ──
    # Build a MusicXML score from detections + staff metadata.  JSON-only
    # input — no image needed; the XML builder works purely from
    # coordinates.  Two response modes:
    #   default                  return XML as application/xml text body
    #   ?format=json             return a JSON envelope { xml, summary }
    #
    # Request body (application/json):
    #     {
    #       "staves": [
    #         { "part_id": "P1", "staff_in_part": 0,
    #           "top_y": 363, "bot_y": 425,
    #           "left_x": 0, "right_x": 2549,
    #           "line_positions": [363, 378, 393, 410, 425],
    #           "line_spacing": 15.5,
    #           "crop_y1": 305, "crop_x1": 137,
    #           "detections": [ {"class": ..., "cx": ..., "cy": ..., ...}, ... ]
    #         }, ...
    #       ],
    #       "num_parts": 1,                  # optional, inferred from part_ids
    #       "instrument_name": "Clarinet",   # optional
    #       "midi_program": 72,              # optional
    #       "divisions": null,               # optional, auto if null
    #       "embed_coordinates": true        # optional
    #     }
    @app.route('/xml_build', methods=['POST'])
    def xml_build_endpoint():
        import tempfile
        try:
            if not request.is_json:
                return jsonify({
                    'error': 'bad_content_type',
                    'message': 'application/json body required.'
                }), 400
            body = request.get_json() or {}

            try:
                processed_score, page_detections = _build_pseudo_score(body)
            except (ValueError, KeyError, TypeError) as ve:
                return jsonify({
                    'error': 'bad_request',
                    'message': f'Could not build score from JSON: {ve}',
                }), 400

            from build_musicxml_v2 import build_score_xml_v2

            # Write to a temp .xml then read back — build_score_xml_v2
            # takes an output_path, not a file handle.
            with tempfile.NamedTemporaryFile(suffix='.xml', delete=False,
                                              mode='w') as tmp:
                tmp_path = tmp.name
            try:
                build_score_xml_v2(
                    processed_score   = processed_score,
                    page_detections   = page_detections,
                    output_path       = tmp_path,
                    instrument_name   = body.get('instrument_name', 'Clarinet'),
                    midi_program      = int(body.get('midi_program', 72)),
                    divisions         = body.get('divisions'),
                    embed_coordinates = bool(body.get('embed_coordinates', True)),
                )
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    xml_text = f.read()
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            if request.args.get('format') == 'json':
                return jsonify({
                    'status':  'ok',
                    'xml':     xml_text,
                    'summary': {
                        'n_staves':     len(page_detections.all_staves),
                        'n_detections': sum(len(sd.detections)
                                            for sd in page_detections.all_staves),
                        'num_parts':    processed_score.num_parts,
                    },
                })
            return Response(xml_text, mimetype='application/xml')
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': 'xml_error',
                            'message': str(e),
                            'type': type(e).__name__}), 500

    # ── GET /health ──
    @app.route('/health', methods=['GET'])
    def health_endpoint():
        return jsonify({
            'status': 'ok',
            'pipeline': {
                'model_path':        str(MODEL_PATH),
                'model_exists':      MODEL_PATH.exists(),
                'output_base_dir':   str(OUTPUT_BASE_DIR),
                'max_image_mb':      MAX_IMAGE_MB,
            },
            'last_result': {
                'has_cached_result': _LAST_RESULT is not None,
                'job_id':            _LAST_RESULT.get('job_id') if _LAST_RESULT else None,
            },
        })

    # ── Catch-all 413 for oversized uploads ──
    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({
            'error':   'image_too_large',
            'message': f'Image exceeds {MAX_IMAGE_MB} MB limit.'
        }), 413

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print('OMR Real API')
    print('-' * 60)
    print(f'  Model           : {MODEL_PATH}')
    print(f'  Model exists    : {MODEL_PATH.exists()}')
    print(f'  Output base dir : {OUTPUT_BASE_DIR}')
    print(f'  Max image size  : {MAX_IMAGE_MB} MB')
    print(f'  Listening on    : http://{HOST}:{PORT}')
    print('-' * 60)
    print('Endpoints:')
    print('  POST /process                  — upload image, get all outputs')
    print('  GET  /rectified                — last rectified PNG')
    print('  GET  /detections               — last detections JSON')
    print('  GET  /xml                      — last MusicXML')
    print('  GET  /full                     — last full response')
    print('  GET  /health                   — readiness check')
    print('  ── per-step (independent of /process cache) ──')
    print('  POST /staff_detect/unet        — staff lines via U-Net')
    print('  POST /staff_detect/classical   — staff lines via classical detector')
    print('  POST /note_align               — batch refinement of detections')
    print('  POST /note_align/click         — single-click refinement (editor UI)')
    print('  POST /xml_build                — build MusicXML from detections')

    if not MODEL_PATH.exists():
        print('\nWARNING: Model file not found.  Set MODEL_PATH env variable')
        print('         or place the .pt file at the expected path.')
        print('         The server will start anyway but /process will fail.')

    app = create_app()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
