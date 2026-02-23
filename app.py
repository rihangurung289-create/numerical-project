"""
Entry point for the Physics & Chemistry problem-solving web application.

This file defines a very small Flask application with:
- a root route that serves the main HTML page
- an example Physics/Chemistry problem endpoint
- a SymPy-based derivative solver endpoint

The goal is to provide a clean, extendable structure without advanced logic yet.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError
import pytesseract
import sympy as sp


app = Flask(
    __name__,
    static_folder=".",  # Serve CSS/JS from the project root
    static_url_path="",
    template_folder=".",  # Render index.html from the project root
)


@dataclass
class ProblemPayload:
    """Lightweight representation of the incoming problem description."""

    subject: str
    topic: str
    problem_text: str


@dataclass
class PhysicsSolution:
    """Container for a detected physics numerical solution."""

    problem_type: str
    formula_used: str
    steps: List[str]
    final_answer: str


def parse_problem_payload(raw: Dict[str, Any]) -> ProblemPayload:
    """
    Extract and normalize the expected fields from the JSON request body.

    This function does only minimal validation. More detailed checks can be
    added later (e.g., allowed subjects/topics, length limits, etc.).
    """
    subject = (raw.get("subject") or "").strip().lower()
    topic = (raw.get("topic") or "").strip()
    problem_text = (raw.get("problemText") or "").strip()

    return ProblemPayload(subject=subject, topic=topic, problem_text=problem_text)


def extract_text_from_image_file(file_storage) -> str:
    """
    Extract text from an uploaded image using Tesseract OCR.

    The caller is expected to handle any exceptions raised here and convert
    them into API-friendly error responses.
    """
    # Read the file content into memory.
    image_bytes = file_storage.read()
    if not image_bytes:
        raise ValueError("Uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc

    try:
        text = pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR engine is not available on the server. "
            "Please install it and ensure it is on the system PATH."
        ) from exc

    return text.strip()


def _find_value_with_keywords(text: str, keywords: List[str]) -> Optional[float]:
    """
    Heuristically associate numbers with nearby keywords (e.g., 'mass', 'kg').

    This is intentionally simple and is meant for structured, numeric problems,
    not arbitrary natural language.
    """
    text_lower = text.lower()
    matches = list(re.finditer(r"\d+(\.\d+)?", text_lower))

    for match in matches:
        start, end = match.span()
        window_start = max(0, start - 24)
        window_end = min(len(text_lower), end + 24)
        window = text_lower[window_start:window_end]
        if any(keyword in window for keyword in keywords):
            try:
                return float(match.group())
            except ValueError:
                continue
    return None


def _solve_newton_second_law(text: str) -> Optional[PhysicsSolution]:
    """
    Attempt to solve a Newton's second law problem (F = m·a).
    """
    m = _find_value_with_keywords(text, ["mass", " kg", "kilogram"])
    a = _find_value_with_keywords(text, ["acceleration", "accelerates", " m/s^2", " m/s2"])
    f = _find_value_with_keywords(text, ["force", " newton", " n "])

    known = {name: value for name, value in [("m", m), ("a", a), ("F", f)] if value is not None}
    unknowns = [name for name in ("m", "a", "F") if locals()[name.lower() if name != "F" else "f"] is None]

    if len(known) < 2 or len(unknowns) != 1:
        return None

    steps: List[str] = [
        "Recognize this as a Newton's second law problem (F = m·a).",
        f"Identify known quantities: {', '.join(f'{k} = {v}' for k, v in known.items())}.",
    ]

    if "F" in unknowns and m is not None and a is not None:
        result = m * a
        formula = "F = m · a"
        steps.append("Compute F using F = m·a.")
        steps.append(f"F = {m} × {a} = {result}.")
        return PhysicsSolution(
            problem_type="newton_second_law",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )
    if "m" in unknowns and f is not None and a is not None and a != 0:
        result = f / a
        formula = "m = F / a"
        steps.append("Compute m using m = F/a.")
        steps.append(f"m = {f} ÷ {a} = {result}.")
        return PhysicsSolution(
            problem_type="newton_second_law",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )
    if "a" in unknowns and f is not None and m is not None and m != 0:
        result = f / m
        formula = "a = F / m"
        steps.append("Compute a using a = F/m.")
        steps.append(f"a = {f} ÷ {m} = {result}.")
        return PhysicsSolution(
            problem_type="newton_second_law",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )

    return None


def _solve_kinematics(text: str) -> Optional[PhysicsSolution]:
    """
    Attempt to solve a constant-acceleration kinematics problem using
    v = u + a·t and related rearrangements.
    """
    text_lower = text.lower()
    u: Optional[float]

    if "from rest" in text_lower or "starts from rest" in text_lower:
        u = 0.0
    else:
        u = _find_value_with_keywords(
            text,
            ["initial velocity", "u=", "starts with velocity", "speed of"],
        )

    v = _find_value_with_keywords(
        text,
        ["final velocity", "velocity", "speed", "v="],
    )
    a = _find_value_with_keywords(
        text,
        ["acceleration", "accelerates", " m/s^2", " m/s2"],
    )
    t = _find_value_with_keywords(
        text,
        ["time", "second", "s ", "for", "after"],
    )

    # Decide which variable is the unknown by inspecting the question.
    question_segment = text_lower[text_lower.rfind("find") :]
    unknown: Optional[str] = None
    if "acceleration" in question_segment:
        unknown = "a"
    elif "initial velocity" in question_segment or "starting velocity" in question_segment:
        unknown = "u"
    elif "final velocity" in question_segment or "velocity" in question_segment or "speed" in question_segment:
        unknown = "v"
    elif "time" in question_segment or "how long" in question_segment:
        unknown = "t"

    # Fallback: infer unknown as the single missing variable.
    if unknown is None:
        present = {name for name, value in [("u", u), ("v", v), ("a", a), ("t", t)] if value is not None}
        missing = [name for name in ("u", "v", "a", "t") if name not in present]
        if len(missing) == 1:
            unknown = missing[0]

    if unknown is None:
        return None

    steps: List[str] = [
        "Recognize this as a constant-acceleration kinematics problem.",
        f"Identify known quantities: "
        f"{', '.join(f'{name} = {val}' for name, val in [('u', u), ('v', v), ('a', a), ('t', t)] if val is not None)}.",
    ]

    # Solve using v = u + a·t and its rearrangements.
    if unknown == "v" and u is not None and a is not None and t is not None:
        result = u + a * t
        formula = "v = u + a·t"
        steps.append("Use v = u + a·t to find the final velocity.")
        steps.append(f"v = {u} + {a} × {t} = {result}.")
        return PhysicsSolution(
            problem_type="kinematics",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )
    if unknown == "u" and v is not None and a is not None and t is not None:
        result = v - a * t
        formula = "u = v - a·t"
        steps.append("Rearrange v = u + a·t to u = v - a·t.")
        steps.append(f"u = {v} - {a} × {t} = {result}.")
        return PhysicsSolution(
            problem_type="kinematics",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )
    if unknown == "a" and v is not None and u is not None and t is not None and t != 0:
        result = (v - u) / t
        formula = "a = (v - u) / t"
        steps.append("Rearrange v = u + a·t to a = (v - u)/t.")
        steps.append(f"a = ({v} - {u}) ÷ {t} = {result}.")
        return PhysicsSolution(
            problem_type="kinematics",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )
    if unknown == "t" and v is not None and u is not None and a is not None and a != 0:
        result = (v - u) / a
        formula = "t = (v - u) / a"
        steps.append("Rearrange v = u + a·t to t = (v - u)/a.")
        steps.append(f"t = ({v} - {u}) ÷ {a} = {result}.")
        return PhysicsSolution(
            problem_type="kinematics",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )

    return None


def _solve_work_energy(text: str) -> Optional[PhysicsSolution]:
    """
    Attempt to solve a work-energy problem using W = F·d.
    """
    w = _find_value_with_keywords(text, ["work", " joule", " j "])
    f = _find_value_with_keywords(text, ["force", " newton", " n "])
    d = _find_value_with_keywords(text, ["distance", "displacement", " m "])

    known = {name: value for name, value in [("W", w), ("F", f), ("d", d)] if value is not None}
    unknowns = [name for name in ("W", "F", "d") if locals()[name.lower() if name != "W" else "w"] is None]

    if len(known) < 2 or len(unknowns) != 1:
        return None

    steps: List[str] = [
        "Recognize this as a work-energy problem using W = F·d.",
        f"Identify known quantities: {', '.join(f'{k} = {v}' for k, v in known.items())}.",
    ]

    if "W" in unknowns and f is not None and d is not None:
        result = f * d
        formula = "W = F · d"
        steps.append("Compute work using W = F·d.")
        steps.append(f"W = {f} × {d} = {result}.")
        return PhysicsSolution(
            problem_type="work_energy",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )
    if "F" in unknowns and w is not None and d is not None and d != 0:
        result = w / d
        formula = "F = W / d"
        steps.append("Compute force using F = W/d.")
        steps.append(f"F = {w} ÷ {d} = {result}.")
        return PhysicsSolution(
            problem_type="work_energy",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )
    if "d" in unknowns and w is not None and f is not None and f != 0:
        result = w / f
        formula = "d = W / F"
        steps.append("Compute distance using d = W/F.")
        steps.append(f"d = {w} ÷ {f} = {result}.")
        return PhysicsSolution(
            problem_type="work_energy",
            formula_used=formula,
            steps=steps,
            final_answer=str(result),
        )

    return None


def solve_physics_problem(text: str) -> Optional[PhysicsSolution]:
    """
    High-level entry point that tries to detect and solve supported physics problems.

    Currently supports:
    - Kinematics equations (constant acceleration)
    - Newton's second law
    - Work-energy theorem (W = F·d)
    """
    text_stripped = text.strip()
    if not text_stripped:
        return None

    # Try specific detectors. The order reflects a simple heuristic and can be
    # adjusted later if needed.
    for solver in (_solve_kinematics, _solve_newton_second_law, _solve_work_energy):
        result = solver(text_stripped)
        if result is not None:
            return result

    return None


def compute_derivative(expression: str) -> Dict[str, Any]:
    """
    Compute the derivative of the given expression with respect to x.

    The function accepts a string expression and returns a small dictionary
    containing:
    - steps: a list of high-level textual steps
    - final_answer: the derivative as a SymPy string

    This keeps the solver logic decoupled from the Flask route.
    """
    x = sp.symbols("x")

    # SymPy parses the string into a symbolic expression.
    # Only very light validation is done here; invalid expressions
    # are handled by the caller via exceptions.
    expr = sp.sympify(expression, locals={"x": x})

    derivative = sp.diff(expr, x)
    simplified = sp.simplify(derivative)

    steps: List[str] = [
        f"Interpret the input as a function of x: f(x) = {sp.srepr(expr)}.",
        "Differentiate f(x) with respect to x using symbolic rules.",
        "Simplify the resulting expression.",
    ]

    return {
        "steps": steps,
        "final_answer": str(simplified),
    }


@app.route("/", methods=["GET"])
def index() -> Any:
    """
    Serve the main HTML page.

    Flask is configured to treat the project root as the template directory,
    so it can render the provided index.html directly.
    """
    return render_template("index.html")


@app.route("/api/solve", methods=["POST"])
def solve_problem() -> Any:
    """
    Endpoint for submitting Physics/Chemistry problems, either as text or image.

    - If called with JSON, expects a body compatible with ProblemPayload.
    - If called with multipart/form-data and an 'image' field, runs OCR to
      extract the problem text before solving.
    """
    data: Dict[str, Any]

    # Image-based workflow using multipart/form-data.
    if "image" in (request.files or {}):
        image_file = request.files["image"]
        if not image_file or image_file.filename == "":
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "No image file was uploaded.",
                    }
                ),
                400,
            )

        try:
            extracted_text = extract_text_from_image_file(image_file)
        except (ValueError, RuntimeError) as exc:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": str(exc),
                    }
                ),
                400,
            )

        if not extracted_text:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Could not detect any readable text in the image.",
                    }
                ),
                400,
            )

        # Subject and topic may optionally be provided as regular form fields.
        subject = (request.form.get("subject") or "").strip().lower()
        topic = (request.form.get("topic") or "").strip()

        data = {
            "subject": subject,
            "topic": topic,
            "problemText": extracted_text,
        }
    else:
        # JSON-based workflow (original behavior).
        data = request.get_json(force=True, silent=True) or {}
    payload = parse_problem_payload(data)

    if not payload.problem_text:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Please include a short problem description in 'problemText' or upload an image.",
                }
            ),
            400,
        )

    subject_label = "Physics" if payload.subject == "physics" else "Chemistry"

    # Default response if no specialized solver is available.
    base_response: Dict[str, Any] = {
        "ok": True,
        "message": (
            f"Received a {subject_label} problem on '{payload.topic or 'general'}'. "
            "Specialized solvers will add more detail when available."
        ),
    }

    if payload.subject == "physics":
        physics_solution = solve_physics_problem(payload.problem_text)
        if physics_solution is not None:
            base_response.update(
                {
                    "problem_type": physics_solution.problem_type,
                    "formula_used": physics_solution.formula_used,
                    "steps": physics_solution.steps,
                    "final_answer": physics_solution.final_answer,
                }
            )

    return jsonify(base_response)


@app.route("/solve", methods=["POST"])
def solve_derivative() -> Any:
    """
    Solve the derivative of a mathematical expression with respect to x.

    The request body is expected to be JSON with:
    - expression: math expression string, e.g. "x**2 + 3*x"

    The response returns JSON with:
    - steps: list of high-level text descriptions
    - final_answer: the derivative as a string
    """
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    expression = (data.get("expression") or "").strip()

    if not expression:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Please include a non-empty 'expression' field.",
                }
            ),
            400,
        )

    try:
        result = compute_derivative(expression)
    except Exception as exc:  # SymPy raises various subclasses; treat uniformly here.
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"Could not parse or differentiate the expression: {exc}",
                }
            ),
            400,
        )

    return jsonify(
        {
            "ok": True,
            "steps": result["steps"],
            "final_answer": result["final_answer"],
        }
    )


@app.route("/<path:filename>", methods=["GET"])
def serve_static_file(filename: str) -> Any:
    """
    Serve static assets (CSS, JS, images) from the project root.

    This keeps the project structure simple: index.html, style.css, script.js,
    and app.py can all live side-by-side.
    """
    return send_from_directory(app.static_folder, filename)


if __name__ == "__main__":
    # Debug mode is convenient during development but should be disabled in production.
    app.run(debug=True)

