
# main.py — full featured build (fresh rules, sanitized kwargs, typed fill, consistent preview/download)
from __future__ import annotations

import io
import json
import csv
import operator
import inspect
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, Tuple, Union

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


APP_TITLE = "Timeseries Post-Processing App"
UPLOAD_KEY = "upload_module"
RULES_KEY = "add_columns_module"
APP_CFG_VERSION = 1  # global config version


# =============================================================================
# Session init & header
# =============================================================================
def get_default_upload_cfg() -> Dict[str, Any]:
    return {
        "source_type": None,          # "csv" | "parquet" | "xlsx"
        "encoding": "utf-8",
        "csv": {
            "sep_mode": "auto",       # "auto" | "," | ";" | "\t" | "space"
            "decimal": ".",           # "." or ","
            "header_row": 0,          # int or None
        },
        "xlsx": {
            "sheet_name": None,       # str or None
            "header_row": 0,          # int or None
        },
        "parse": {
            "datetime_col": None,     # str | None
            "value_col": None,        # str | None (for quick preview)
            "set_index_to_datetime": True,
        }
    }


def get_default_rules_cfg() -> Dict[str, Any]:
    return {
        "version": 1,
        "rules": []  # list of rule dicts
    }


def init_state():
    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "version": APP_CFG_VERSION,
            UPLOAD_KEY: get_default_upload_cfg(),
            RULES_KEY: get_default_rules_cfg(),
        }
    else:
        st.session_state.cfg.setdefault("version", APP_CFG_VERSION)
        st.session_state.cfg.setdefault(UPLOAD_KEY, get_default_upload_cfg())
        st.session_state.cfg.setdefault(RULES_KEY, get_default_rules_cfg())

    if "df" not in st.session_state:
        st.session_state.df = None
    if "file_meta" not in st.session_state:
        st.session_state.file_meta = {"name": None, "size": 0}

    if "rules_runtime" not in st.session_state:
        st.session_state.rules_runtime = {
            "preview_rows": 100,
            "apply_on_copy": True,
            "force_overwrite": True,  # global toggle to allow all overwrites on this run
        }


def ui_app_header():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Module 1: Upload data • Module 2: Add columns (JSON-driven rules). "
               "Preview/Full runs always use the freshest rules; downloads match what you see.")


# =============================================================================
# Sidebar: Global Config Import/Export (single JSON for all modules)
# =============================================================================
def ui_global_config_sidebar():
    st.sidebar.header("App Configuration")
    st.sidebar.caption("Export or load all module settings (upload + rules).")

    # Export
    cfg_export = json.dumps(st.session_state.cfg, ensure_ascii=False, indent=2)
    st.sidebar.download_button(
        "⬇️ Download full config (.json)",
        data=cfg_export,
        file_name="timeseries_app_config.json",
        mime="application/json",
        key="dl_full_cfg",
    )

    # Import
    up_cfg = st.sidebar.file_uploader(
        "⬆️ Load full config (.json)", type=["json"], key="full_cfg_uploader"
    )
    if up_cfg is not None:
        try:
            text = up_cfg.read().decode("utf-8")
            new_cfg = json.loads(text)
            if not isinstance(new_cfg, dict):
                raise ValueError("Invalid config file.")
            if UPLOAD_KEY not in new_cfg or RULES_KEY not in new_cfg:
                raise ValueError(f"JSON must contain '{UPLOAD_KEY}' and '{RULES_KEY}'.")
            st.session_state.cfg = {
                "version": new_cfg.get("version", APP_CFG_VERSION),
                UPLOAD_KEY: new_cfg.get(UPLOAD_KEY, get_default_upload_cfg()),
                RULES_KEY: new_cfg.get(RULES_KEY, get_default_rules_cfg()),
            }
            st.sidebar.success("Configuration loaded.")
        except Exception as e:
            st.sidebar.error(f"Failed to load configuration: {e}")


# =============================================================================
# Module 1: Upload
# =============================================================================
def sniff_delimiter(sample: bytes) -> str:
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample.decode("utf-8", errors="ignore"))
        return dialect.delimiter
    except Exception:
        return ","  # sensible default


def read_csv_file(file_bytes: bytes, encoding: str, sep_mode: str, decimal: str, header_row: Optional[int]) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if sep_mode == "auto":
        sample = file_bytes[:4096]
        sep = sniff_delimiter(sample)
    elif sep_mode == "space":
        sep = r"\s+"
    elif sep_mode == "\t":
        sep = "\t"
    else:
        sep = sep_mode  # "," or ";"

    df = pd.read_csv(
        buf,
        sep=sep,
        engine="python" if sep in [r"\s+", "\t"] else "c",
        encoding=encoding,
        decimal=decimal,
        header=header_row
    )
    return df


def read_parquet_file(file_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    return pd.read_parquet(buf)


def list_xlsx_sheets(file_bytes: bytes) -> List[str]:
    buf = io.BytesIO(file_bytes)
    xls = pd.ExcelFile(buf)
    return xls.sheet_names


def read_xlsx_file(file_bytes: bytes, sheet_name: Optional[str], header_row: Optional[int]) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    df = pd.read_excel(buf, sheet_name=sheet_name, header=header_row)
    if isinstance(df, dict):
        first_key = list(df.keys())[0]
        df = df[first_key]
    return df


def apply_parse_options(df: pd.DataFrame, parse_cfg: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()
    dt_col = parse_cfg.get("datetime_col")
    set_idx = parse_cfg.get("set_index_to_datetime", True)

    if dt_col and dt_col in df.columns:
        try:
            df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", infer_datetime_format=True, utc=False)
        except Exception:
            pass

        if set_idx and pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df = df.set_index(dt_col).sort_index()

    return df


def ui_upload_module():
    cfg = st.session_state.cfg[UPLOAD_KEY]

    with st.expander("1) Upload data", expanded=True):
        left, right = st.columns([1, 1])

        with right:
            st.subheader("Config (Module)")
            st.caption("Tip: use the sidebar to export/import the whole app config at once.")
            with st.expander("View current upload config", expanded=False):
                st.code(json.dumps(cfg, ensure_ascii=False, indent=2), language="json")

        with left:
            st.subheader("File")
            up = st.file_uploader("Upload a file", type=["csv", "parquet", "xlsx", "xls"], key="file_uploader_main")

            source_type = None
            if up is not None and up.type:
                name = up.name.lower()
                if name.endswith(".csv"):
                    source_type = "csv"
                elif name.endswith(".parquet"):
                    source_type = "parquet"
                elif name.endswith(".xlsx") or name.endswith(".xls"):
                    source_type = "xlsx"
            cfg["source_type"] = source_type

            if source_type == "csv":
                st.markdown("**CSV options**")
                csv_cols = st.columns([1, 1, 1])
                sep_mode = csv_cols[0].selectbox(
                    "Delimiter",
                    options=["auto", ",", ";", "\t", "space"],
                    index=["auto", ",", ";", "\t", "space"].index(cfg["csv"].get("sep_mode", "auto")),
                    key="csv_sep_mode"
                )
                cfg["csv"]["sep_mode"] = sep_mode

                encoding = csv_cols[1].text_input("Encoding", value=cfg.get("encoding", "utf-8"), key="csv_encoding")
                cfg["encoding"] = encoding

                decimal = csv_cols[2].selectbox(
                    "Decimal",
                    options=[".", ","],
                    index=[".", ","].index(cfg["csv"].get("decimal", ".")),
                    key="csv_decimal"
                )
                cfg["csv"]["decimal"] = decimal

                header_row = st.number_input(
                    "Header row (0-based; -1 for no header)",
                    value=int(cfg["csv"].get("header_row", 0) if cfg["csv"].get("header_row", 0) is not None else 0),
                    step=1,
                    min_value=-1
                )
                cfg["csv"]["header_row"] = None if header_row < 0 else int(header_row)

            elif source_type == "xlsx":
                st.markdown("**Excel options**")
                sheets = []
                chosen_sheet = cfg["xlsx"].get("sheet_name")
                header_row = cfg["xlsx"].get("header_row", 0)
                if up is not None:
                    try:
                        file_bytes = up.getvalue()
                        sheets = list_xlsx_sheets(file_bytes)
                        if chosen_sheet not in sheets and sheets:
                            chosen_sheet = sheets[0]
                    except Exception as e:
                        st.warning(f"Could not list sheets: {e}")

                ex_cols = st.columns([2, 1])
                sheet_name = ex_cols[0].selectbox(
                    "Sheet name",
                    options=sheets if sheets else [chosen_sheet] if chosen_sheet else [],
                    index=0 if sheets else 0,
                    key="xlsx_sheet_name"
                ) if sheets else chosen_sheet
                cfg["xlsx"]["sheet_name"] = sheet_name

                header_row_in = ex_cols[1].number_input(
                    "Header row (0-based; -1 for no header)",
                    value=int(header_row if header_row is not None else 0),
                    step=1,
                    min_value=-1
                )
                cfg["xlsx"]["header_row"] = None if header_row_in < 0 else int(header_row_in)

            elif source_type == "parquet":
                st.info("No options required for Parquet.")

            read_btn = st.button("Read file", type="primary", disabled=(up is None), key="btn_read_file")
            if read_btn and up is not None:
                try:
                    file_bytes = up.getvalue()
                    st.session_state.file_meta = {"name": up.name, "size": len(file_bytes)}

                    if source_type == "csv":
                        df = read_csv_file(
                            file_bytes=file_bytes,
                            encoding=cfg.get("encoding", "utf-8"),
                            sep_mode=cfg["csv"].get("sep_mode", "auto"),
                            decimal=cfg["csv"].get("decimal", "."),
                            header_row=cfg["csv"].get("header_row", 0),
                        )
                    elif source_type == "parquet":
                        df = read_parquet_file(file_bytes)
                    elif source_type == "xlsx":
                        df = read_xlsx_file(
                            file_bytes=file_bytes,
                            sheet_name=cfg["xlsx"].get("sheet_name"),
                            header_row=cfg["xlsx"].get("header_row", 0),
                        )
                    else:
                        raise ValueError("Unsupported or unknown file type.")

                    st.session_state.df = df
                    st.success(f"Loaded: **{up.name}** with shape {df.shape}")

                except Exception as e:
                    st.session_state.df = None
                    st.error(f"Failed to read file: {e}")

            clear = st.button("Clear data", type="secondary", key="btn_clear_data")
            if clear:
                st.session_state.df = None
                st.session_state.file_meta = {"name": None, "size": 0}
                st.info("Cleared uploaded data from memory.")

        df = st.session_state.df
        if df is not None:
            st.divider()
            st.subheader("Parse options & preview")

            columns = list(df.columns)
            parse_cfg = cfg["parse"]

            c1, c2, c3 = st.columns([1.3, 1.3, 1])
            datetime_col = c1.selectbox(
                "Datetime column (optional, will parse + set as index if selected)",
                options=["<none>"] + columns,
                index=(columns.index(parse_cfg.get("datetime_col")) + 1) if parse_cfg.get("datetime_col") in columns else 0,
                key="parse_datetime_col"
            )
            parse_cfg["datetime_col"] = None if datetime_col == "<none>" else datetime_col

            value_col = c2.selectbox(
                "Value column for quick preview",
                options=["<none>"] + columns,
                index=(columns.index(parse_cfg.get("value_col")) + 1) if parse_cfg.get("value_col") in columns else 0,
                key="parse_value_col"
            )
            parse_cfg["value_col"] = None if value_col == "<none>" else value_col

            set_index_to_dt = c3.toggle(
                "Set index to datetime",
                value=bool(parse_cfg.get("set_index_to_datetime", True)),
                key="parse_set_index"
            )
            parse_cfg["set_index_to_datetime"] = bool(set_index_to_dt)

            df_preview = apply_parse_options(df, parse_cfg)

            st.caption("Data preview (first 100 rows):")
            st.dataframe(df_preview.head(100), use_container_width=True)

            if parse_cfg["value_col"] and parse_cfg["value_col"] in df_preview.columns:
                try:
                    fig = px.line(
                        df_preview.reset_index(),
                        x=df_preview.reset_index().columns[0],
                        y=parse_cfg["value_col"],
                        title="Quick preview"
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=350)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"Plot preview not available: {e}")

            st.download_button(
                "Download preview as CSV",
                data=df_preview.to_csv(index=True).encode("utf-8"),
                file_name="preview_parsed.csv",
                mime="text/csv",
                key="dl_preview_csv"
            )


# =============================================================================
# Module 2: Rules — core logic
# =============================================================================
def new_rule_template(rule_type: str, name_hint: str = "newcol") -> Dict[str, Any]:
    if rule_type == "condition":
        return {
            "type": "condition",
            "name": f"{name_hint}_cond",
            "input": {
                "left": {"kind": "column", "value": ""},
                "op": ">",
                "right": {"kind": "const", "dtype": "float", "value": 0.0}
            },
            "missing_policy": "propagate",
            "fill_value": None,
            "cast": "bool",
            "allow_overwrite": False
        }
    if rule_type == "arithmetic":
        return {
            "type": "arithmetic",
            "name": f"{name_hint}_arith",
            "input": {
                "left": {"kind": "column", "value": ""},
                "op": "-",
                "right": {"kind": "column", "value": ""}
            },
            "missing_policy": "propagate",
            "fill_value": None,
            "cast": "float",
            "allow_overwrite": False
        }
    if rule_type == "function":
        return {
            "type": "function",
            "name": f"{name_hint}_func",
            "func": "cp",
            "args": [
                {"name": "series", "kind": "column", "value": ""}  # first arg = series
            ],
            "kwargs": {},
            "missing_policy": "propagate",
            "fill_value": None,
            "cast": None,
            "allow_overwrite": False
        }
    raise ValueError("Unknown rule type")


# ---- Function registry (safe implementations)
@dataclass
class FuncSpec:
    func: Callable
    arg_kinds: List[str]          # e.g., ["series"] for first positional
    defaults: Dict[str, Any]
    doc: str
    output_dtype_hint: Optional[str] = None  # "float"|"int"|"bool"|None


def fn_cp(series: pd.Series, *args, normalize: bool = False, clip: Optional[Tuple[float, float]] = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce").astype(float)
    for a in args:
        if isinstance(a, pd.Series):
            out = out + pd.to_numeric(a, errors="coerce").astype(float)
        else:
            out = out + float(a)
    if normalize:
        mu = out.mean()
        sd = out.std(ddof=0) or 1.0
        out = (out - mu) / sd
    if clip is not None and (clip[0] is not None or clip[1] is not None):
        low, high = clip
        out = out.clip(lower=low, upper=high)
    return out


def fn_zscore(series: pd.Series, window: int = 30, min_periods: Optional[int] = None) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    if min_periods is None:
        min_periods = max(1, min(window, len(series)))
    rolling = series.rolling(window=window, min_periods=min_periods)
    return (series - rolling.mean()) / (rolling.std(ddof=0) + 1e-12)


def fn_clip(series: pd.Series, low: Optional[float] = None, high: Optional[float] = None) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.clip(lower=low, upper=high)


def fn_pct_change(series: pd.Series, periods: int = 1) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.pct_change(periods=periods)


def fn_roll_mean(series: pd.Series, window: int = 7, min_periods: Optional[int] = 1) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.rolling(window=window, min_periods=min_periods).mean()


FUNC_REGISTRY: Dict[str, FuncSpec] = {
    "cp": FuncSpec(
        func=fn_cp,
        arg_kinds=["series"],
        defaults={"normalize": False, "clip": None},
        doc="Custom combine function: start from series, optional normalize/clip. (*args not exposed as named params.)",
        output_dtype_hint="float",
    ),
    "zscore": FuncSpec(
        func=fn_zscore,
        arg_kinds=["series"],
        defaults={"window": 30, "min_periods": None},
        doc="Rolling z-score of the series.",
        output_dtype_hint="float",
    ),
    "clip": FuncSpec(
        func=fn_clip,
        arg_kinds=["series"],
        defaults={"low": None, "high": None},
        doc="Clip the series between low/high.",
        output_dtype_hint=None,
    ),
    "pct_change": FuncSpec(
        func=fn_pct_change,
        arg_kinds=["series"],
        defaults={"periods": 1},
        doc="Percent change over given periods.",
        output_dtype_hint="float",
    ),
    "roll_mean": FuncSpec(
        func=fn_roll_mean,
        arg_kinds=["series"],
        defaults={"window": 7, "min_periods": 1},
        doc="Rolling mean with window and min_periods.",
        output_dtype_hint="float",
    ),
}


# ---- Missing policy (typed) & casting
def _coerce_missing(series: pd.Series, policy: str, fill_value: Any, cast_hint: Optional[str] = None) -> pd.Series:
    """
    Apply missing policy with smart typing for fill values:
    - numeric series or cast_hint in {"int","float"} -> numeric fill
    - boolean series or cast_hint == "bool" -> boolean fill
    - otherwise: use as-is
    """
    if policy == "propagate":
        return series

    if policy == "false_zero":
        if pd.api.types.is_bool_dtype(series) or cast_hint == "bool":
            return series.fillna(False)
        return series.fillna(0)

    if policy == "fill":
        fv = fill_value

        if cast_hint == "bool":
            if isinstance(fv, str):
                s = fv.strip().lower()
                if s in {"1", "true", "t", "yes", "y"}:
                    fv = True
                elif s in {"0", "false", "f", "no", "n", ""}:
                    fv = False
                else:
                    fv = False
            else:
                fv = bool(fv)
            return series.fillna(fv)

        if pd.api.types.is_numeric_dtype(series) or cast_hint in {"int", "float"}:
            if isinstance(fv, str):
                try:
                    fv = float(fv)
                except Exception:
                    fv = 0.0
            elif fv is None:
                fv = 0.0
            return series.fillna(fv)

        return series.fillna("" if fv is None else fv)

    return series


def _apply_cast(series: pd.Series, cast: Optional[str]) -> pd.Series:
    if not cast:
        return series
    if cast == "bool":
        return series.astype("boolean").astype(bool)
    if cast == "int":
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    if cast == "float":
        return pd.to_numeric(series, errors="coerce").astype(float)
    return series


# ---- Operand helper
def _get_operand(df: pd.DataFrame, spec: Dict[str, Any]) -> Union[pd.Series, Any]:
    kind = spec.get("kind")
    val = spec.get("value")
    if kind == "column":
        if val not in df.columns:
            raise ValueError(f"Column '{val}' not in dataframe.")
        return df[val]
    if kind == "const":
        dtype = spec.get("dtype", "float")
        if dtype == "float":
            try:
                return float(val) if val is not None else np.nan
            except Exception:
                return np.nan
        if dtype == "int":
            try:
                return int(float(val)) if val not in (None, "") else 0
            except Exception:
                return 0
        if dtype == "str":
            return str(val) if val is not None else ""
        if dtype == "list":
            if isinstance(val, str):
                return [v.strip() for v in val.split(",")]
            return list(val) if val is not None else []
        return val
    raise ValueError("Operand 'kind' must be 'column' or 'const'.")


# ---- Evaluators
def eval_condition_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
    left = _get_operand(df, rule["input"]["left"])
    right = _get_operand(df, rule["input"]["right"])
    op = rule["input"]["op"]

    if op in {">", ">=", "<", "<=", "==", "!="}:
        if not isinstance(left, pd.Series):
            raise ValueError("Left side must be a column/series for comparisons.")
        left_c = left
        right_c = right.reindex(left_c.index) if isinstance(right, pd.Series) else right
        pyop = {
            ">": operator.gt, ">=": operator.ge,
            "<": operator.lt, "<=": operator.le,
            "==": operator.eq, "!=": operator.ne
        }[op]
        out = pyop(left_c, right_c)
    elif op in {"isna", "notna"}:
        if not isinstance(left, pd.Series):
            raise ValueError("isna/notna requires left to be a column/series.")
        out = left.isna() if op == "isna" else left.notna()
    elif op in {"in", "not in"}:
        if isinstance(right, pd.Series):
            out = left.isin(right)
        else:
            right_list = right if isinstance(right, (list, tuple, set)) else [right]
            out = left.isin(right_list)
        if op == "not in":
            out = ~out
    else:
        raise ValueError(f"Unsupported operator '{op}' for condition rule.")

    out = out.astype("boolean")
    out = _coerce_missing(out, rule.get("missing_policy", "propagate"), rule.get("fill_value"), cast_hint=rule.get("cast"))
    out = _apply_cast(out, rule.get("cast"))
    return out


def eval_arithmetic_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
    op = rule["input"]["op"]
    pyop = {
        "+": operator.add, "-": operator.sub, "*": operator.mul,
        "/": operator.truediv, "//": operator.floordiv,
        "%": operator.mod, "**": operator.pow
    }.get(op)
    if pyop is None:
        raise ValueError(f"Unsupported arithmetic operator '{op}'.")

    left = _get_operand(df, rule["input"]["left"])
    right = _get_operand(df, rule["input"]["right"])

    if not isinstance(left, pd.Series):
        raise ValueError("Left operand must be a dataframe column (select one).")

    lnum = pd.to_numeric(left, errors="coerce")

    if isinstance(right, pd.Series):
        rnum = pd.to_numeric(right, errors="coerce").reindex(lnum.index)
    else:
        rnum = float(right)

    if op in {"/", "//", "%"}:
        if isinstance(rnum, pd.Series):
            zero_mask = (rnum == 0)
            safe_r = rnum.mask(zero_mask, np.nan) if zero_mask.any() else rnum
        else:
            safe_r = (np.nan if rnum == 0 else rnum)

        if op == "/":
            out = lnum / safe_r
        elif op == "//":
            out = lnum // safe_r
        else:
            out = lnum % safe_r

        if isinstance(out, pd.Series):
            out = out.replace([np.inf, -np.inf], np.nan)
    else:
        out = pyop(lnum, rnum if isinstance(rnum, pd.Series) else rnum)
        if isinstance(out, pd.Series):
            out = out.replace([np.inf, -np.inf], np.nan)

    out = _coerce_missing(out, rule.get("missing_policy", "propagate"), rule.get("fill_value"), cast_hint=rule.get("cast"))
    out = _apply_cast(out, rule.get("cast"))
    return out


# ---- Function param helpers
def _allowed_kwargs_for_func(fn: Callable) -> set:
    """Return set of valid keyword parameter names for fn (excluding first 'series')."""
    sig = inspect.signature(fn)
    names = []
    for i, (name, p) in enumerate(sig.parameters.items()):
        if name == "self":
            continue
        if i == 0 and name.lower() in ("series", "s", "x"):
            continue
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            names.append(name)
    return set(names)


def _sig_params_for_func(fn: Callable):
    """
    Return (positional_param_names, kw_param_defaults) for a function.
    Used to seed defaults for the UI picker.
    """
    sig = inspect.signature(fn)
    pos_names = []
    kw_defaults = {}
    for i, (name, p) in enumerate(sig.parameters.items()):
        if name == "self":
            continue
        if i == 0 and name.lower() in ("series", "s", "x"):
            continue
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            if p.default is inspect._empty:
                pos_names.append(name)
            else:
                pos_names.append(name)
                kw_defaults[name] = p.default
        elif p.kind == p.KEYWORD_ONLY:
            kw_defaults[name] = (None if p.default is inspect._empty else p.default)
    return pos_names, kw_defaults


def eval_function_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
    fname = rule.get("func")
    if fname not in FUNC_REGISTRY:
        raise ValueError(f"Function '{fname}' is not available.")
    spec = FUNC_REGISTRY[fname]

    # Build positional args (only first positional is allowed: the series)
    args_cfg = rule.get("args", [])
    if not args_cfg:
        raise ValueError("Function requires a series argument.")
    first = _get_operand(df, args_cfg[0])
    if spec.arg_kinds and spec.arg_kinds[0] == "series" and not isinstance(first, pd.Series):
        raise ValueError(f"First argument to '{fname}' must be a column/series.")
    built_args = [first]

    # kwargs with defaults then user overrides; filter to allowed
    kwargs = dict(spec.defaults)
    kwargs.update(rule.get("kwargs", {}))
    allowed_kw = _allowed_kwargs_for_func(spec.func)
    kwargs = {k: v for k, v in kwargs.items() if k in allowed_kw}

    out = spec.func(*built_args, **kwargs)
    if not isinstance(out, pd.Series):
        out = pd.Series(out, index=df.index)

    out = _coerce_missing(out, rule.get("missing_policy", "propagate"), rule.get("fill_value"), cast_hint=rule.get("cast"))
    out = _apply_cast(out, rule.get("cast"))
    return out


def evaluate_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> pd.Series:
    rtype = rule.get("type")
    if rtype == "condition":
        return eval_condition_rule(df, rule)
    if rtype == "arithmetic":
        return eval_arithmetic_rule(df, rule)
    if rtype == "function":
        return eval_function_rule(df, rule)
    raise ValueError(f"Unknown rule type '{rtype}'.")


# ---- Completeness & validation
def is_complete_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> Tuple[bool, str]:
    rtype = rule.get("type")
    name = (rule.get("name") or "").strip()
    if not name:
        return False, "Name is required."

    if rtype == "condition":
        inp = rule.get("input", {})
        left = inp.get("left", {})
        op = inp.get("op")
        right = inp.get("right", {})
        if left.get("kind") != "column" or not left.get("value"):
            return False, "Pick a left column."
        if left["value"] not in df.columns:
            return False, f"Left column '{left['value']}' not in dataframe."
        if op not in {">", ">=", "<", "<=", "==", "!=", "isna", "notna", "in", "not in"}:
            return False, "Choose an operator."
        if op not in {"isna", "notna"}:
            rk = right.get("kind", "const")
            if rk == "column":
                if not right.get("value"):
                    return False, "Pick a right column or choose constant."
                if right["value"] not in df.columns:
                    return False, f"Right column '{right['value']}' not in dataframe."
        return True, "OK"

    if rtype == "arithmetic":
        inp = rule.get("input", {})
        left = inp.get("left", {})
        op = inp.get("op")
        right = inp.get("right", {})
        if left.get("kind") != "column" or not left.get("value"):
            return False, "Pick a left column."
        if left["value"] not in df.columns:
            return False, f"Left column '{left['value']}' not in dataframe."
        if op not in {"+", "-", "*", "/", "//", "%", "**"}:
            return False, "Choose a valid arithmetic operator."
        rk = right.get("kind", "column")
        if rk == "column":
            if not right.get("value"):
                return False, "Pick a right column or choose constant."
            if right["value"] not in df.columns:
                return False, f"Right column '{right['value']}' not in dataframe."
        return True, "OK"

    if rtype == "function":
        fname = rule.get("func")
        if not fname or fname not in FUNC_REGISTRY:
            return False, "Pick a function."
        args = rule.get("args", [])
        if not args:
            return False, "Add at least one argument (first must be a column)."
        first = args[0]
        if first.get("kind") != "column" or not first.get("value"):
            return False, "First argument must be a column."
        if first["value"] not in df.columns:
            return False, f"Column '{first['value']}' not in dataframe."
        return True, "OK"

    return False, f"Unknown rule type '{rtype}'."


def validate_rule(df: pd.DataFrame, rule: Dict[str, Any]) -> Tuple[bool, str]:
    name = (rule.get("name") or "").strip()
    if not name:
        return False, "Name is required."
    if not all(c.isalnum() or c in "_-" for c in name):
        return False, "Name must be alphanumeric with _ or -."

    complete, msg = is_complete_rule(df, rule)
    if not complete:
        return False, f"Incomplete: {msg}"

    try:
        if rule.get("type") == "function":
            spec = FUNC_REGISTRY[rule["func"]]
            allowed_kw = _allowed_kwargs_for_func(spec.func)
            bad_keys = [k for k in rule.get("kwargs", {}).keys() if k not in allowed_kw]
            if bad_keys:
                return False, f"Invalid parameters for {rule['func']}: {', '.join(bad_keys)}"
        if rule.get("type") == "arithmetic":
            _ = eval_arithmetic_rule(df.head(5), rule)
        elif rule.get("type") == "condition":
            _ = eval_condition_rule(df.head(5), rule)
        elif rule.get("type") == "function":
            _ = eval_function_rule(df.head(5), rule)
        else:
            return False, f"Unknown rule type '{rule.get('type')}'."
        return True, "OK"
    except Exception as e:
        return False, str(e)


def apply_rules(df: pd.DataFrame, rules: List[Dict[str, Any]], preview_rows: Optional[int] = None) -> Tuple[pd.DataFrame, List[Tuple[int, str, Optional[str]]]]:
    """
    Apply rules in order. Returns (new_df, logs) where logs contain (idx, name, error or None).
    Skips incomplete rules (logs as 'Skipped (incomplete)').
    """
    target = df.head(preview_rows).copy() if preview_rows else df.copy()
    logs: List[Tuple[int, str, Optional[str]]] = []
    for i, rule in enumerate(rules):
        name = rule.get("name", f"rule_{i}")
        complete, msg = is_complete_rule(target, rule)
        if not complete:
            logs.append((i, name, f"Skipped (incomplete): {msg}"))
            continue
        try:
            series = evaluate_rule(target, rule)
            if (name in target.columns) and not rule.get("allow_overwrite", False):
                raise ValueError(f"Column '{name}' already exists. Enable 'Allow overwrite' to replace.")
            target[name] = series
            logs.append((i, name, None))
        except Exception as e:
            logs.append((i, name, str(e)))
    return target, logs


# ---- Rules materialization for runs
def _sanitize_rule_kwargs(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure kwargs for function rules match the selected function signature; keep only 1 positional (series)."""
    if rule.get("type") != "function":
        return rule
    fname = rule.get("func")
    spec = FUNC_REGISTRY.get(fname)
    if not spec:
        return rule
    allowed = _allowed_kwargs_for_func(spec.func)
    cur = dict(spec.defaults)
    user = rule.get("kwargs", {}) or {}
    user = {k: v for k, v in user.items() if k in allowed}
    cur.update(user)
    rule["kwargs"] = cur
    args = rule.get("args", [])
    if not args:
        args = [{"name": "series", "kind": "column", "value": ""}]
    if len(args) > 1:
        args = args[:1]
    rule["args"] = args
    return rule


def _materialize_rules(rules: List[Dict[str, Any]], force_overwrite: bool = False) -> List[Dict[str, Any]]:
    """Deep-copy rules, sanitize kwargs, and optionally set allow_overwrite=True; also dedup by column name (keep last)."""
    try:
        tmp = json.loads(json.dumps(rules))
    except Exception:
        tmp = [json.loads(json.dumps(r)) for r in rules]  # fallback

    mat = []
    for r in tmp:
        r = _sanitize_rule_kwargs(r)
        if force_overwrite:
            r["allow_overwrite"] = True
        mat.append(r)

    # Deduplicate by name: keep last occurrence
    seen = set()
    dedup = []
    for r in reversed(mat):
        nm = (r.get("name") or "").strip().lower()
        if nm and nm not in seen:
            dedup.append(r)
            seen.add(nm)
    dedup.reverse()
    return dedup


# =============================================================================
# Module 2: Rules — UI
# =============================================================================
def _ui_rule_condition(df: pd.DataFrame, rule: Dict[str, Any], i: int):
    cols = st.columns([2, 1, 2, 1.2, 1.2, 1])
    all_cols = [""] + list(df.columns)

    left_val = rule["input"]["left"].get("value", "")
    rule["input"]["left"]["kind"] = "column"
    rule["input"]["left"]["value"] = cols[0].selectbox(
        "Left column", options=all_cols, index=all_cols.index(left_val) if left_val in all_cols else 0, key=f"cond_left_{i}"
    )

    ops = [">", ">=", "<", "<=", "==", "!=", "isna", "notna", "in", "not in"]
    rule["input"]["op"] = cols[1].selectbox("Operator", options=ops, index=ops.index(rule["input"].get("op", ">")), key=f"cond_op_{i}")

    right_kind = cols[2].selectbox(
        "Right kind", options=["const", "column"], index=["const", "column"].index(rule["input"]["right"].get("kind", "const")), key=f"cond_rightkind_{i}"
    )
    rule["input"]["right"]["kind"] = right_kind
    if right_kind == "column":
        r_val = rule["input"]["right"].get("value", "")
        rule["input"]["right"]["value"] = cols[2].selectbox(
            "Right column", options=all_cols, index=all_cols.index(r_val) if r_val in all_cols else 0, key=f"cond_rightcol_{i}"
        )
        rule["input"]["right"].pop("dtype", None)
    else:
        dtype_opts = ["float", "int", "str", "list"]
        dtype = cols[2].selectbox(
            "Const dtype", options=dtype_opts, index=dtype_opts.index(rule["input"]["right"].get("dtype", "float")), key=f"cond_dtype_{i}"
        )
        rule["input"]["right"]["dtype"] = dtype
        val_default = rule["input"]["right"].get("value", "")
        val_text = cols[2].text_input("Const value (comma-sep for list)", value=str(val_default), key=f"cond_const_{i}")
        rule["input"]["right"]["value"] = val_text

    pol = ["propagate", "false_zero", "fill"]
    rule["missing_policy"] = cols[3].selectbox("Missing policy", options=pol, index=pol.index(rule.get("missing_policy", "propagate")), key=f"cond_miss_{i}")
    rule["fill_value"] = cols[4].text_input("Fill value (if policy=fill)", value=str(rule.get("fill_value", "")), key=f"cond_fill_{i}")
    cast_opts = ["bool", "int", "float", None]
    cast_label = ["bool", "int", "float", "None"]
    current_cast = rule.get("cast", "bool")
    cast_idx = cast_label.index("None" if current_cast in (None, "None") else current_cast)
    set_cast = cols[5].selectbox("Cast", options=cast_label, index=cast_idx, key=f"cond_cast_{i}")
    rule["cast"] = None if set_cast == "None" else set_cast


def _ui_rule_arithmetic(df: pd.DataFrame, rule: Dict[str, Any], i: int):
    cols = st.columns([2, 1, 2, 1.2, 1.2, 1])
    all_cols = [""] + list(df.columns)

    lval = rule["input"]["left"].get("value", "")
    rule["input"]["left"]["kind"] = "column"
    rule["input"]["left"]["value"] = cols[0].selectbox(
        "Left column", options=all_cols, index=all_cols.index(lval) if lval in all_cols else 0, key=f"arith_left_{i}"
    )

    ops = ["+", "-", "*", "/", "//", "%", "**"]
    rule["input"]["op"] = cols[1].selectbox("Operator", options=ops, index=ops.index(rule["input"].get("op", "-")), key=f"arith_op_{i}")

    rk = cols[2].selectbox(
        "Right kind", options=["column", "const"], index=["column", "const"].index(rule["input"]["right"].get("kind", "column")), key=f"arith_rk_{i}"
    )
    rule["input"]["right"]["kind"] = rk
    if rk == "column":
        rv = rule["input"]["right"].get("value", "")
        rule["input"]["right"]["value"] = cols[2].selectbox(
            "Right column", options=all_cols, index=all_cols.index(rv) if rv in all_cols else 0, key=f"arith_rightcol_{i}"
        )
        rule["input"]["right"].pop("dtype", None)
    else:
        dtype_opts = ["float", "int"]
        dtype = cols[2].selectbox(
            "Const dtype", options=dtype_opts, index=dtype_opts.index(rule["input"]["right"].get("dtype", "float")), key=f"arith_dtype_{i}"
        )
        rule["input"]["right"]["dtype"] = dtype
        txt = cols[2].text_input("Const value", value=str(rule["input"]["right"].get("value", 0)), key=f"arith_const_{i}")
        try:
            rule["input"]["right"]["value"] = float(txt) if dtype == "float" else int(float(txt))
        except Exception:
            rule["input"]["right"]["value"] = 0.0 if dtype == "float" else 0

    pol = ["propagate", "false_zero", "fill"]
    rule["missing_policy"] = cols[3].selectbox(
        "Missing policy", options=pol, index=pol.index(rule.get("missing_policy", "propagate")), key=f"arith_miss_{i}"
    )
    rule["fill_value"] = cols[4].text_input("Fill value (if policy=fill)", value=str(rule.get("fill_value", "")), key=f"arith_fill_{i}")

    cast_opts = ["float", "int", None]
    cast_label = ["float", "int", "None"]
    current_cast = rule.get("cast", "float")
    cast_idx = cast_label.index("None" if current_cast in (None, "None") else current_cast)
    set_cast = cols[5].selectbox("Cast", options=cast_label, index=cast_idx, key=f"arith_cast_{i}")
    rule["cast"] = None if set_cast == "None" else set_cast


def _ui_rule_function(df: pd.DataFrame, rule: Dict[str, Any], i: int):
    cols = st.columns([2, 2, 2, 2, 2])
    fnames = list(FUNC_REGISTRY.keys())
    fname = cols[0].selectbox(
        "Function", options=fnames,
        index=fnames.index(rule.get("func", "cp")) if rule.get("func", "cp") in fnames else 0,
        key=f"func_name_{i}"
    )

    # Detect function change & sanitize kwargs
    prev_func = rule.get("_prev_func", rule.get("func", fname))
    rule["func"] = fname
    spec = FUNC_REGISTRY.get(fname)
    allowed_kw = _allowed_kwargs_for_func(spec.func) if spec else set()

    if fname != prev_func:
        old_kwargs = dict(rule.get("kwargs", {}))
        new_kwargs = dict(spec.defaults) if spec else {}
        for k, v in old_kwargs.items():
            if k in allowed_kw:
                new_kwargs[k] = v
        rule["kwargs"] = new_kwargs
        rule["_prev_func"] = fname

    # Positional (only first = series)
    st.markdown("**Arguments**")
    args_list: List[Dict[str, Any]] = rule.get("args", [])
    if not args_list:
        args_list.append({"name": "series", "kind": "column", "value": ""})
    if len(args_list) > 1:
        args_list = args_list[:1]
    rule["args"] = args_list

    all_cols = [""] + list(df.columns)
    arow = st.columns([1.1, 2.4, 2.0, 1.0])
    args_list[0]["name"] = arow[0].text_input("Positional #1", value=args_list[0].get("name", "series"), key=f"f_argname_{i}_0")
    args_list[0]["kind"] = "column"
    aval = args_list[0].get("value", "")
    args_list[0]["value"] = arow[1].selectbox("Series column", options=all_cols, index=all_cols.index(aval) if aval in all_cols else 0, key=f"f_argc_{i}_0")

    # Keyword parameters by signature
    st.markdown("**Keyword parameters**")
    if spec:
        _, kw_defaults = _sig_params_for_func(spec.func)
        cur_kwargs = dict(spec.defaults)
        cur_kwargs.update(rule.get("kwargs", {}))
        cur_kwargs = {k: v for k, v in cur_kwargs.items() if k in allowed_kw}
        rule["kwargs"] = cur_kwargs

        already = set(cur_kwargs.keys())
        available = sorted([k for k in allowed_kw if k not in already])

        picker_cols = st.columns([2, 1])
        picked = picker_cols[0].selectbox("Add parameter", options=["<select>"] + available, index=0, key=f"kw_add_{i}")
        add_btn = picker_cols[1].button("Add", key=f"kw_add_btn_{i}", disabled=(picked == "<select>"))

        if add_btn and picked != "<select>":
            default_val = kw_defaults.get(picked, None)
            cur_kwargs[picked] = default_val
            rule["kwargs"] = cur_kwargs
            st.experimental_rerun()

        new_kwargs = {}
        for k, v in cur_kwargs.items():
            if isinstance(v, bool):
                new_kwargs[k] = st.toggle(k, value=bool(v), key=f"f_kw_{i}_{k}")
            elif isinstance(v, int):
                new_kwargs[k] = st.number_input(k, value=int(v), step=1, key=f"f_kw_{i}_{k}")
            elif isinstance(v, float):
                new_kwargs[k] = st.number_input(k, value=float(v), step=0.1, key=f"f_kw_{i}_{k}")
            elif v is None:
                new_kwargs[k] = st.text_input(k, value="", key=f"f_kw_{i}_{k}") or None
            elif isinstance(v, (list, tuple)):
                entry = st.text_input(k + " (comma-sep)", value=",".join(map(str, v)), key=f"f_kw_{i}_{k}")
                new_kwargs[k] = [x.strip() for x in entry.split(",")] if entry else []
            elif isinstance(v, dict):
                entry = st.text_area(k + " (JSON)", value=json.dumps(v), key=f"f_kw_{i}_{k}")
                try:
                    new_kwargs[k] = json.loads(entry) if entry.strip() else {}
                except Exception:
                    new_kwargs[k] = v
            else:
                new_kwargs[k] = st.text_input(k, value=str(v), key=f"f_kw_{i}_{k}")
        rule["kwargs"] = new_kwargs

        st.caption("Only the first positional argument is the **series**. Add valid keyword parameters using the picker.")

    # Policies & casting
    rule["missing_policy"] = cols[1].selectbox(
        "Missing policy", options=["propagate", "false_zero", "fill"],
        index=["propagate", "false_zero", "fill"].index(rule.get("missing_policy", "propagate")),
        key=f"f_miss_{i}"
    )
    rule["fill_value"] = cols[2].text_input("Fill value (if policy=fill)", value=str(rule.get("fill_value", "")), key=f"f_fill_{i}")
    cast_opts = ["float", "int", "bool", None]
    cast_label = ["float", "int", "bool", "None"]
    current_cast = rule.get("cast", None)
    cast_idx = cast_label.index("None" if current_cast in (None, "None") else current_cast)
    set_cast = cols[3].selectbox("Cast", options=cast_label, index=cast_idx, key=f"f_cast_{i}")
    rule["cast"] = None if set_cast == "None" else set_cast


def _render_run_result(out_df: pd.DataFrame, logs: List[Tuple[int, str, Optional[str]]], is_preview: bool, show_logs: bool):
    kind = "Preview" if is_preview else "Result"
    st.subheader(f"{kind} — output")
    ok_count = sum(1 for _, _, err in logs if err is None)
    err_count = sum(1 for _, _, err in logs if err is not None and not str(err).startswith("Skipped"))
    skip_count = sum(1 for _, _, err in logs if err is not None and str(err).startswith("Skipped"))
    st.write(f"Rules applied: ✅ {ok_count}  •  ⚠️ {err_count}  •  ⏭️ Skipped: {skip_count}")

    if err_count or skip_count:
        with st.expander("Errors / Skipped", expanded=show_logs):
            for idx, name, err in logs:
                if err:
                    st.write(f"- Rule #{idx} **{name}** → {err}")

    st.dataframe(out_df.head(200), use_container_width=True)
    st.download_button(
        "Download output CSV",
        data=out_df.to_csv(index=True).encode("utf-8"),
        file_name=("preview_with_newcols.csv" if is_preview else "data_with_newcols.csv"),
        mime="text/csv",
        key=("dl_preview_out" if is_preview else "dl_full_out")
    )


def ui_rules_module():
    df = st.session_state.df
    cfg = st.session_state.cfg[RULES_KEY]
    rules: List[Dict[str, Any]] = cfg.get("rules", [])
    rt = st.session_state.rules_runtime

    with st.expander("2) Add columns (Rules)", expanded=True):
        top_left, top_mid, top_right = st.columns([1, 1, 1])

        with top_left:
            st.subheader("Rules")
            add_type = st.selectbox("New rule type", options=["condition", "arithmetic", "function"], key="rule_new_type")
            if st.button("➕ Add rule", type="primary", key="btn_add_rule"):
                rules.append(new_rule_template(add_type))
                cfg["rules"] = rules

        with top_mid:
            st.subheader("Preview / Run")
            rt["preview_rows"] = st.slider("Preview N rows", min_value=20, max_value=2000, step=20, value=rt.get("preview_rows", 100), key="slider_preview_rows")
            rt["apply_on_copy"] = st.toggle("Apply on a copy (safer)", value=rt.get("apply_on_copy", True), key="toggle_apply_on_copy")

        with top_right:
            st.subheader("Run options")
            st.caption("Configure run controls below via the Run buttons (force overwrite & logs).")

        st.markdown("---")

        if df is None:
            st.info("Upload data in Module 1 to define and preview rules.")
            return

        # Rule list UI
        for i, rule in enumerate(list(rules)):
            with st.container(border=True):
                head_cols = st.columns([4, 3, 3, 2, 2, 2])
                rule["name"] = head_cols[0].text_input("Name", value=rule.get("name", ""), key=f"rule_name_{i}")

                prev_type = rule.get("_prev_type", rule.get("type", "condition"))
                selected_type = head_cols[1].selectbox(
                    "Type",
                    options=["condition", "arithmetic", "function"],
                    index=["condition", "arithmetic", "function"].index(rule.get("type", "condition")),
                    key=f"rule_type_{i}"
                )

                if selected_type != prev_type:
                    preserved_name = rule.get("name", "")
                    preserved_over = rule.get("allow_overwrite", False)
                    fresh = new_rule_template(selected_type, name_hint=preserved_name or "newcol")
                    fresh["name"] = preserved_name or fresh["name"]
                    fresh["allow_overwrite"] = preserved_over
                    fresh["_prev_type"] = selected_type
                    rules[i] = fresh
                    rule = rules[i]
                rule["_prev_type"] = selected_type

                rule["allow_overwrite"] = head_cols[2].checkbox("Allow overwrite", value=rule.get("allow_overwrite", False), key=f"rule_over_{i}")
                move_up = head_cols[3].button("↑", key=f"rule_up_{i}")
                move_dn = head_cols[4].button("↓", key=f"rule_dn_{i}")
                dup_btn = head_cols[5].button("⧉", key=f"rule_dup_{i}")

                if move_up and i > 0:
                    rules[i - 1], rules[i] = rules[i], rules[i - 1]
                if move_dn and i < len(rules) - 1:
                    rules[i + 1], rules[i] = rules[i], rules[i + 1]
                if dup_btn:
                    rules.insert(i + 1, json.loads(json.dumps(rule)))  # deep copy

                del_col = st.columns([1, 9])[0]
                if del_col.button("Delete", key=f"rule_del_{i}"):
                    rules.pop(i)
                    cfg["rules"] = rules
                    st.stop()

                st.markdown("")

                if rule["type"] == "condition":
                    _ui_rule_condition(df, rule, i)
                elif rule["type"] == "arithmetic":
                    _ui_rule_arithmetic(df, rule, i)
                else:
                    _ui_rule_function(df, rule, i)

                ok, msg = validate_rule(df, rule)
                vcol1, vcol2 = st.columns([1, 3])
                if ok:
                    vcol1.success("Valid")
                    try:
                        prev_series = evaluate_rule(df.head(rt["preview_rows"]), rule)
                        if pd.api.types.is_bool_dtype(prev_series):
                            pct_true = float(prev_series.mean(skipna=True)) * 100.0
                            vcol2.write(f"**Preview:** {prev_series.name} — %True ≈ {pct_true:.1f}%")
                            fig = px.bar(x=["True", "False"], y=[pct_true, 100 - pct_true], title="Preview distribution")
                            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=200)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            stats = prev_series.describe(percentiles=[0.25, 0.5, 0.75])
                            vcol2.write(f"**Preview:** {prev_series.name} — mean={stats.get('mean', np.nan):.4g}, min={stats.get('min', np.nan):.4g}, max={stats.get('max', np.nan):.4g}")
                            fig = px.line(prev_series.reset_index(), x=prev_series.reset_index().columns[0], y=prev_series.name, title="Preview sparkline")
                            fig.update_layout(margin=dict(l=10, r=10, t=35, b=10), height=220)
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        vcol2.info(f"Preview error: {e}")
                else:
                    vcol1.warning("Incomplete" if msg.startswith("Incomplete") else "Invalid")
                    vcol2.write(msg)

                st.markdown("---")

        cfg["rules"] = rules

        # Run buttons and toggles (unique keys)
        run_cols = st.columns([1, 1, 2, 2])
        run_preview = run_cols[0].button("Run on preview", key="btn_run_preview")
        run_apply = run_cols[1].button("Run on full data", type="primary", key="btn_run_apply")
        rt["force_overwrite"] = run_cols[2].toggle("Force overwrite existing columns", value=rt.get("force_overwrite", True), key="toggle_force_overwrite")
        show_logs = run_cols[3].toggle("Show logs", value=False, key="toggle_show_logs")

        # Use a fresh snapshot of rules from session state for each run, then materialize
        if run_preview:
            rules_current = st.session_state.cfg[RULES_KEY].get("rules", [])
            rules_to_run = _materialize_rules(rules_current, force_overwrite=rt.get("force_overwrite", False))
            out_df, logs = apply_rules(st.session_state.df, rules_to_run, preview_rows=rt["preview_rows"])
            _render_run_result(out_df, logs, is_preview=True, show_logs=show_logs)

        if run_apply:
            base = st.session_state.df.copy() if rt["apply_on_copy"] else st.session_state.df
            rules_current = st.session_state.cfg[RULES_KEY].get("rules", [])
            rules_to_run = _materialize_rules(rules_current, force_overwrite=rt.get("force_overwrite", False))
            out_df, logs = apply_rules(base, rules_to_run, preview_rows=None)
            # Always sync session df to output so preview & download reflect latest run
            st.session_state.df = out_df
            _render_run_result(out_df, logs, is_preview=False, show_logs=show_logs)


# =============================================================================
# App main
# =============================================================================
def main():
    init_state()
    ui_app_header()
    ui_global_config_sidebar()
    ui_upload_module()
    ui_rules_module()


if __name__ == "__main__":
    main()
