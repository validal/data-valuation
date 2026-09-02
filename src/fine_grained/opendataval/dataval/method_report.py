"""Generate a concise report of train/eval phases for all DataEvaluator methods.

This scans `opendataval.dataval` for classes that inherit from `DataEvaluator` and
extracts a short summary of what happens in:
- `train_data_values()` ("train" phase)
- `evaluate_data_values()` ("eval" phase)

The report is based on:
- class docstrings (if present)
- method docstrings (if present)
- a lightweight static analysis of method bodies (key calls + line counts)

Usage
-----
python -m opendataval.dataval.method_report --out dataval_method_report.md

Notes
-----
Timing/memory accounting is handled centrally by `DataEvaluator.train()` and the
cached `DataEvaluator.data_values` property (see `opendataval/dataval/api.py`).
This script only describes *what* each evaluator implements in train/eval.
"""

from __future__ import annotations

import argparse
import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class MethodSummary:
	name: str
	lineno: Optional[int]
	end_lineno: Optional[int]
	doc: str
	key_calls: tuple[str, ...]


@dataclass(frozen=True)
class ClassSummary:
	qualname: str
	module_relpath: str
	bases: tuple[str, ...]
	class_doc: str
	train: Optional[MethodSummary]
	eval: Optional[MethodSummary]
	is_abstract: bool


def _base_name(expr: ast.expr) -> str:
	if isinstance(expr, ast.Name):
		return expr.id
	if isinstance(expr, ast.Attribute):
		parts: list[str] = []
		cur: ast.AST = expr
		while isinstance(cur, ast.Attribute):
			parts.append(cur.attr)
			cur = cur.value
		if isinstance(cur, ast.Name):
			parts.append(cur.id)
		return ".".join(reversed(parts))
	if isinstance(expr, ast.Subscript):
		return _base_name(expr.value)
	return expr.__class__.__name__


def _inherits_data_evaluator(class_node: ast.ClassDef) -> bool:
	for base in class_node.bases:
		if _base_name(base).endswith("DataEvaluator"):
			return True
	return False


def _is_abstract_class(class_node: ast.ClassDef) -> bool:
	# Heuristic: inherits ABC/ABCMeta or contains @abstractmethod.
	for base in class_node.bases:
		b = _base_name(base)
		if b.endswith("ABC") or b.endswith("ABCMeta"):
			return True
	for item in class_node.body:
		if isinstance(item, ast.FunctionDef):
			for dec in item.decorator_list:
				if _base_name(dec).endswith("abstractmethod"):
					return True
	return False


def _first_paragraph(text: str) -> str:
	text = (text or "").strip()
	if not text:
		return ""
	parts = [p.strip() for p in text.split("\n\n")]
	return parts[0]


def _collect_key_calls(func_node: ast.FunctionDef, max_items: int = 12) -> tuple[str, ...]:
	"""Collect a stable set of call names (lightweight static signal)."""

	def _call_name(call: ast.Call) -> str:
		f = call.func
		if isinstance(f, ast.Name):
			return f.id
		if isinstance(f, ast.Attribute):
			qual = _base_name(f.value)
			if qual == "self":
				return f"self.{f.attr}"
			if qual in {"np", "numpy", "torch", "sklearn", "scipy", "math"}:
				return f"{qual}.{f.attr}"
			return f.attr
		return f.__class__.__name__

	seen: set[str] = set()
	items: list[str] = []
	for node in ast.walk(func_node):
		if isinstance(node, ast.Call):
			name = _call_name(node)
			if name not in seen:
				seen.add(name)
				items.append(name)
	return tuple(items[:max_items])


def _summarize_method(func_node: ast.FunctionDef) -> MethodSummary:
	doc = ast.get_docstring(func_node) or ""
	return MethodSummary(
		name=func_node.name,
		lineno=getattr(func_node, "lineno", None),
		end_lineno=getattr(func_node, "end_lineno", None),
		doc=_first_paragraph(doc),
		key_calls=_collect_key_calls(func_node),
	)


def _iter_python_files(root: Path) -> Iterable[Path]:
	for dirpath, _, filenames in os.walk(root):
		for fn in filenames:
			if fn.endswith(".py"):
				yield Path(dirpath) / fn


def scan_dataval(dataval_dir: Path) -> list[ClassSummary]:
	summaries: list[ClassSummary] = []

	for py_file in sorted(_iter_python_files(dataval_dir)):
		if "__pycache__" in py_file.parts:
			continue

		rel = py_file.relative_to(dataval_dir.parent)  # opendataval/dataval/...
		try:
			src = py_file.read_text(encoding="utf-8")
		except Exception:
			continue

		try:
			tree = ast.parse(src, filename=str(py_file))
		except SyntaxError:
			continue

		for node in tree.body:
			if not isinstance(node, ast.ClassDef):
				continue
			if not _inherits_data_evaluator(node):
				continue

			class_doc = ast.get_docstring(node) or ""
			bases = tuple(_base_name(b) for b in node.bases)

			train_node: Optional[ast.FunctionDef] = None
			eval_node: Optional[ast.FunctionDef] = None
			for item in node.body:
				if isinstance(item, ast.FunctionDef):
					if item.name == "train_data_values":
						train_node = item
					elif item.name == "evaluate_data_values":
						eval_node = item

			train_sum = _summarize_method(train_node) if train_node else None
			eval_sum = _summarize_method(eval_node) if eval_node else None

			summaries.append(
				ClassSummary(
					qualname=node.name,
					module_relpath=str(rel),
					bases=bases,
					class_doc=_first_paragraph(class_doc),
					train=train_sum,
					eval=eval_sum,
					is_abstract=_is_abstract_class(node),
				)
			)

	summaries.sort(key=lambda s: (s.module_relpath, s.qualname))
	return summaries


def to_markdown(items: list[ClassSummary]) -> str:
	lines: list[str] = []
	lines.append("# DataEvaluator Train/Eval Phase Report")
	lines.append("")
	lines.append("This report is generated by `python -m opendataval.dataval.method_report`.")
	lines.append("")
	lines.append('## What "train" and "eval" mean here')
	lines.append("")
	lines.append("- **Train phase** = the evaluator's `train_data_values()` implementation, executed inside `DataEvaluator.train()`. ")
	lines.append("- **Eval phase** = the evaluator's `evaluate_data_values()` implementation, executed inside the cached `DataEvaluator.data_values` property.")
	lines.append("- The framework measures elapsed seconds (and CPU/GPU memory snapshots) around these calls and stores them in `memory_report` / `memory_report_eval`.")
	lines.append("")

	lines.append("## Methods")
	lines.append("")
	for cs in items:
		lines.append(f"- `{cs.qualname}` ({cs.module_relpath})")
	lines.append("")

	def _is_generic_doc(doc: str) -> bool:
		d = (doc or "").strip().lower()
		return d in {
			"trains model to predict data values.",
			"return data values for each training data point.",
			"return computed data values.",
			"return computed data values",
			"return data values for each training data point",
		}

	def _summary_from_calls(cs: ClassSummary, ms: Optional[MethodSummary], phase: str) -> str:
		if ms is None:
			return "Not implemented."

		calls = set(ms.key_calls)
		bases = set(cs.bases)

		# Special-cases / strong signals
		if cs.qualname == "RandomEvaluator":
			return "No training; returns random scores." if phase == "train" else "Samples random values (uniform)."

		if "compute_marginal_contribution" in calls:
			return "Samples coalitions and accumulates marginal contributions (semivalue/Shapley-style)."

		if cs.qualname == "DataOob":
			if phase == "train":
				return "Trains many bootstrapped models; tracks out-of-bag (OOB) memberships and predictions."
			return "Aggregates per-point loss over models where the point was OOB."

		if "DatasetDistance" in calls or "FeatureCost" in calls or cs.qualname.startswith("Lava"):
			if phase == "train":
				if cs.qualname == "LavaOOBEvaluator":
					return "Computes LAVA-style values via repeated OT on bootstrapped subsets (OOB-style approximation)."
				if cs.qualname == "BatchwiseLavaEvaluator":
					return "Computes LAVA values in batches using embeddings and OT-based distances for scalability."
				return "Computes LAVA values from embeddings using OT/Wasserstein class distances."
			return "Returns the cached OT-based values (with light post-processing)."

		if cs.qualname.startswith("KNNShapley") or "torch.argsort" in calls:
			if phase == "train":
				return "Computes KNN-based Shapley values from embeddings/distances (sorting neighbors and accumulating contributions)."
			return "Returns the computed KNN-Shapley values."

		if cs.qualname.startswith("Influence") or "iter_grad" in calls:
			if phase == "train":
				if "Pool" in calls:
					return "Fits base model(s) and estimates influence via repeated subsampling in parallel."
				if "choice" in calls and "Subset" in calls:
					return "Fits many models on random subsamples and collects per-point losses for influence-style scoring."
				return "Fits base model and computes gradients/HVP terms needed for influence scores."
			return "Computes per-point influence scores from stored gradient/subsample statistics."

		if cs.qualname in {"DVRL", "DVRLShap"} or "DveLoss" in calls:
			if phase == "train":
				return "Trains a value estimator with reinforcement learning to predict per-point utility (DVRL-style)."
			return "Runs the trained value estimator to output per-point values."

		if cs.qualname in {"LeaveOneOut", "LOORemovalRanker"}:
			if phase == "train":
				return "Retrains/evaluates models while leaving out points (or removing in ranked order) to estimate contribution."
			return "Returns the stored leave-one-out based values/ranks."

		if cs.qualname in {"ForgettingEvents"}:
			if phase == "train":
				return "Trains across epochs and counts forgetting events per point (correct→incorrect transitions)."
			return "Returns per-point forgetting-event counts as values."

		if cs.qualname in {"GAVA"}:
			if phase == "train":
				return "Generates values via randomized grouping/aggregation on embeddings (model-less heuristic valuation)."
			return "Returns the cached heuristic values."

		if "fit" in calls and "predict" in calls and ("ModelMixin" in bases or "self.evaluate" in calls):
			if phase == "train":
				if "binomial" in calls and "Subset" in calls:
					return "Trains many models on random Bernoulli-selected subsets; logs OOB/inclusion patterns and losses."
				if "permutation" in calls:
					return "Trains baseline + leave-one-out retrains to estimate each point’s marginal effect on performance."
				return "Trains model(s) and collects per-point losses/metrics needed for valuation."
			if "np.divide" in calls or "np.zeros_like" in calls:
				return "Aggregates stored losses/metrics into a per-point value vector (normalization by counts)."
			return "Returns the computed per-point values (may include normalization)."

		if "self.embeddings" in calls or "ModelLessMixin" in bases:
			return "Computes values from embeddings/distances without training the prediction model." if phase == "train" else "Returns embedding-based values."

		# Fallback
		if phase == "train":
			return "Computes and stores intermediate statistics needed to value each training point."
		return "Returns the final per-point data values from stored statistics."

	def _render_simple(cs: ClassSummary):
		lines.append(f"## {cs.qualname}")
		lines.append("")
		lines.append(f"- **Module:** `{cs.module_relpath}`")
		if cs.class_doc:
			lines.append(f"- **Method:** {cs.class_doc}")

		train_text = ""
		if cs.train and cs.train.doc and not _is_generic_doc(cs.train.doc):
			train_text = cs.train.doc
		else:
			train_text = _summary_from_calls(cs, cs.train, phase="train")

		eval_text = ""
		if cs.eval and cs.eval.doc and not _is_generic_doc(cs.eval.doc):
			eval_text = cs.eval.doc
		else:
			eval_text = _summary_from_calls(cs, cs.eval, phase="eval")

		lines.append(f"- **Train (`train_data_values`)**: {train_text}")
		lines.append(f"- **Eval (`evaluate_data_values`)**: {eval_text}")
		lines.append("")

	for cs in items:
		_render_simple(cs)

	return "\n".join(lines)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--out",
		type=str,
		default="dataval_method_report.md",
		help="Output markdown file path (default: dataval_method_report.md)",
	)
	args = parser.parse_args()

	dataval_dir = Path(__file__).resolve().parent
	items = scan_dataval(dataval_dir)
	md = to_markdown(items)

	out_path = Path(args.out).expanduser().resolve()
	out_path.write_text(md, encoding="utf-8")
	print(f"Wrote {out_path} ({len(items)} classes)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
