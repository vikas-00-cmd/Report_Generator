import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from pathlib import Path

# Optional GUI for input selection
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog, ttk
    TK_AVAILABLE = True
except ImportError:
    tk = None
    TK_AVAILABLE = False

# ── Fix Windows DPI blurriness (must run before any Tk window is created) ──
import sys, ctypes
if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # Per-monitor DPI aware v2
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()   # Fallback
        except Exception:
            pass

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=None, help="Path to CSV dataset file (skips file picker GUI)")
parser.add_argument("--features", default=None, help="Comma-separated feature column names")
parser.add_argument("--target", default=None, help="Target column name")
parser.add_argument("--gui", action="store_true", help="Launch GUI for selecting features and targets")
args = parser.parse_args()


# ─────────────────────────────────────────────
# GUI Step 1 – CSV File Picker
# ─────────────────────────────────────────────

def launch_file_picker() -> str:
    """
    Opens a styled Tkinter window where the user can type or browse
    for a CSV file. Returns the selected file path as a string.
    """
    if not TK_AVAILABLE:
        raise RuntimeError("Tkinter is unavailable. Install tkinter or pass --data <path> to skip the GUI.")

    result = {"path": None}

    root = tk.Tk()
    root.title("Energy Report – Select Dataset")
    root.resizable(False, False)
    root.configure(bg="#f0f4f8")

    # ── Header ──────────────────────────────────
    header = tk.Frame(root, bg="#0078d4", height=48)
    header.pack(fill="x")
    tk.Label(
        header,
        text="📂  Select your CSV dataset",
        bg="#0078d4", fg="white",
        font=("Segoe UI", 13, "bold"),
        pady=10,
    ).pack(side="left", padx=16)

    # ── Body ─────────────────────────────────────
    body = tk.Frame(root, bg="#f0f4f8", padx=20, pady=14)
    body.pack(fill="x")

    tk.Label(body, text="Dataset file path:", bg="#f0f4f8",
             font=("Segoe UI", 10)).pack(anchor="w", pady=(0, 5))

    row = tk.Frame(body, bg="#f0f4f8")
    row.pack(fill="x")

    path_var = tk.StringVar(value="")
    path_entry = tk.Entry(row, textvariable=path_var,
                          font=("Segoe UI", 10), relief="solid", bd=1)
    path_entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 8))

    def browse():
        # Hide the launcher so it doesn't appear dimmed behind the OS dialog
        root.withdraw()
        chosen = filedialog.askopenfilename(
            parent=root,
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.deiconify()      # Bring launcher back after dialog closes
        root.lift()           # Ensure it comes to front
        root.focus_force()
        if chosen:
            path_var.set(chosen)
            path_entry.icursor("end")   # Scroll entry to end so filename is visible

    tk.Button(
        row, text="Browse…", command=browse,
        bg="#e1ecf7", fg="#0078d4",
        font=("Segoe UI", 10, "bold"),
        relief="solid", bd=1, padx=10, pady=4, cursor="hand2",
    ).pack(side="left")

    # ── Status label (shown on error) ────────────
    status_var = tk.StringVar()
    tk.Label(body, textvariable=status_var, bg="#f0f4f8",
             fg="#c0392b", font=("Segoe UI", 9)).pack(anchor="w", pady=(5, 0))

    # ── Load button ───────────────────────────────
    def on_load():
        p = path_var.get().strip()
        if not p:
            status_var.set("⚠  Please enter or browse to a CSV file.")
            return
        fp = Path(p)
        if not fp.exists():
            status_var.set(f"⚠  File not found: {fp}")
            return
        if fp.suffix.lower() != ".csv":
            status_var.set("⚠  Please select a .csv file.")
            return
        result["path"] = str(fp)
        root.destroy()

    tk.Button(
        root, text="Load & Continue →", command=on_load,
        bg="#0078d4", fg="white",
        font=("Segoe UI", 11, "bold"),
        relief="flat", padx=14, pady=8, cursor="hand2",
        activebackground="#005fa3", activeforeground="white",
    ).pack(pady=(4, 14))

    # Allow Enter key to trigger load
    root.bind("<Return>", lambda _e: on_load())

    root.mainloop()

    if result["path"] is None:
        raise RuntimeError("No CSV file was selected. Exiting.")

    return result["path"]


# ─────────────────────────────────────────────
# Resolve the data file
# ─────────────────────────────────────────────

if args.data:
    # CLI path provided – skip the file picker
    DATA_FILE = Path(args.data)
    if not DATA_FILE.exists():
        DATA_FILE = Path(__file__).resolve().parent / args.data
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find data file at {args.data}. "
            f"Tried: {Path(args.data).resolve()} and {Path(__file__).resolve().parent / args.data}."
        )
else:
    # No --data supplied → always open the graphical file picker
    DATA_FILE = Path(launch_file_picker())


# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────

try:
    data = pd.read_csv(DATA_FILE)
except pd.errors.EmptyDataError:
    raise ValueError(
        f"Data file '{DATA_FILE}' is empty. Please provide a CSV with a header row and at least one data row."
    )

if data.shape[1] == 0:
    raise ValueError(
        f"Data file '{DATA_FILE}' has no columns. Ensure the CSV has a header row and at least one column."
    )

print("Dataset Preview:")
print(data.head())
print("\nColumns:", list(data.columns))

# Normalize column names for case-insensitive matching
column_lower_to_actual = {col.lower(): col for col in data.columns}


def resolve_column(name, kind):
    key = name.strip().lower()
    if key in column_lower_to_actual:
        return column_lower_to_actual[key]
    from difflib import get_close_matches
    close = get_close_matches(key, column_lower_to_actual.keys(), n=1, cutoff=0.6)
    if close:
        resolved = column_lower_to_actual[close[0]]
        print(f"Auto-mapped {kind} '{name}' to existing column '{resolved}'.")
        return resolved
    raise ValueError(f"{kind} column '{name}' not found. Available columns: {list(data.columns)}")


def parse_feature_columns(raw):
    columns = [col.strip() for col in raw.split(",") if col.strip()]
    if not columns:
        raise ValueError("Feature columns are required and cannot be empty.")
    return [resolve_column(col, "Feature") for col in columns]


def parse_target_columns(raw):
    columns = [col.strip() for col in raw.split(",") if col.strip()]
    if not columns:
        raise ValueError("Target column is required and cannot be empty.")
    return [resolve_column(col, "Target") for col in columns]


# ─────────────────────────────────────────────
# GUI Step 2 – Column Selector
# ─────────────────────────────────────────────

def select_columns_with_tkinter(columns):
    if not TK_AVAILABLE:
        raise RuntimeError("Tkinter is unavailable. Install tkinter or run without --gui.")

    selected_result = {"features": None, "targets": None}

    def on_submit():
        feature_indices = list(feature_listbox.curselection())
        target_indices  = list(target_listbox.curselection())
        if not feature_indices:
            messagebox.showerror("Input Required", "Select at least one feature column.")
            return
        if not target_indices:
            messagebox.showerror("Input Required", "Select at least one target column.")
            return
        selected_result["features"] = [columns[i] for i in feature_indices]
        selected_result["targets"]  = [columns[i] for i in target_indices]
        root.destroy()

    root = tk.Tk()
    root.title("Select Features and Targets")
    root.geometry("700x460")
    root.configure(bg="#f0f4f8")

    # Header
    header = tk.Frame(root, bg="#0078d4", height=48)
    header.pack(fill="x")
    tk.Label(
        header,
        text="🔬  Choose feature and target columns",
        bg="#0078d4", fg="white",
        font=("Segoe UI", 13, "bold"),
        pady=10,
    ).pack(side="left", padx=16)

    # Show selected file
    tk.Label(
        root,
        text=f"Dataset: {DATA_FILE.name}",
        bg="#f0f4f8", fg="#555",
        font=("Segoe UI", 9),
    ).pack(anchor="w", padx=18, pady=(8, 0))

    frame = tk.Frame(root, bg="#f0f4f8")
    frame.pack(fill="both", expand=True, padx=14, pady=6)

    for side, label_text, box_ref_key in [
        ("left",  "Feature columns (Ctrl+click for multi-select)", "feature"),
        ("right", "Target columns  (Ctrl+click for multi-select)", "target"),
    ]:
        col_frame = tk.Frame(frame, bg="#f0f4f8")
        col_frame.pack(side=side, fill="both", expand=True, padx=6)
        tk.Label(col_frame, text=label_text, bg="#f0f4f8",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(anchor="w")

        lb = tk.Listbox(
            col_frame, selectmode="extended", exportselection=0,
            font=("Segoe UI", 10),
            relief="solid", bd=1,
            selectbackground="#0078d4", selectforeground="white",
        )
        for col in columns:
            lb.insert("end", col)
        lb.pack(fill="both", expand=True, pady=4)

        if box_ref_key == "feature":
            feature_listbox = lb
        else:
            target_listbox = lb

    tk.Label(
        root,
        text="Hold Ctrl (⌘ on Mac) + click to select multiple columns.",
        bg="#f0f4f8", fg="#888", font=("Segoe UI", 9),
    ).pack()

    tk.Button(
        root, text="Run Report →", command=on_submit,
        bg="#0078d4", fg="white",
        font=("Segoe UI", 11, "bold"),
        relief="flat", padx=14, pady=8, cursor="hand2",
        activebackground="#005fa3", activeforeground="white",
    ).pack(pady=12)

    root.mainloop()

    if selected_result["features"] is None or selected_result["targets"] is None:
        raise RuntimeError("GUI input was cancelled.")
    return selected_result["features"], selected_result["targets"]


# ─────────────────────────────────────────────
# Select Features & Targets
# ─────────────────────────────────────────────

if args.gui or not (args.features and args.target):
    # Use GUI column selector whenever --gui is set OR when either arg is missing
    feature_cols, target_cols = select_columns_with_tkinter(list(data.columns))
else:
    feature_cols = parse_feature_columns(args.features)
    target_cols  = parse_target_columns(args.target)

for t in target_cols:
    if t in feature_cols:
        raise ValueError(f"Target column '{t}' cannot also be a feature column.")

print(f"\nSelected features: {feature_cols}")
print(f"Selected target(s): {target_cols}\n")

X = data[feature_cols]
y = data[target_cols]

# Convert datetime-like and numeric-string columns to numeric values for modeling

def _coerce_column_to_numeric(series, label):
    if pd.api.types.is_numeric_dtype(series):
        return series

    # Only attempt datetime parsing if the column is already datetime-typed
    # OR if a meaningful portion of its string values actually parse as datetimes.
    # This prevents numeric/categorical columns from being silently turned into NaT.
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed_datetime = pd.to_datetime(series, errors='coerce', utc=True)
        parsed_datetime = parsed_datetime.dt.tz_localize(None)
        return parsed_datetime.astype('int64') // 10**9

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        # Probe a sample: only treat as datetime if >80% of non-null values parse
        sample = series.dropna().head(50)
        probe = pd.to_datetime(sample, errors='coerce', utc=True)
        if probe.notna().mean() > 0.8:
            parsed_datetime = pd.to_datetime(series, errors='coerce', utc=True)
            if parsed_datetime.notna().all():
                parsed_datetime = parsed_datetime.dt.tz_localize(None)
                return parsed_datetime.astype('int64') // 10**9

        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().all():
            return numeric

    raise ValueError(
        f"Column '{label}' contains non-numeric values and cannot be used for regression. "
        "Select numeric columns or preprocess the data before running the report."
    )

X = X.apply(lambda s: _coerce_column_to_numeric(s, s.name))
y = y.apply(lambda s: _coerce_column_to_numeric(s, s.name))

# Drop rows with NaNs in selected columns
before = len(data)
joined = pd.concat([X, y], axis=1).dropna()
if len(joined) == 0:
    raise ValueError(
        "No rows remain after dropping NaNs in selected feature and target columns. "
        "Please choose non-missing columns, impute missing values, or clean your dataset."
    )
X = joined[feature_cols]
y = joined[target_cols]
print(f"Using {len(X)} rows after dropping NaNs from selected columns (from {before} original rows).")


# ─────────────────────────────────────────────
# Train ML Model
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression() if y.shape[1] == 1 else MultiOutputRegressor(LinearRegression())
model.fit(X_train, y_train)

predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions, multioutput="uniform_average")
print(f"\nMean Absolute Error (uniform average across targets): {error:.4f}")


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

documents_dir = Path.home() / "Documents"
documents_dir.mkdir(parents=True, exist_ok=True)
image_paths = []

for idx, tname in enumerate(target_cols):
    plt.figure()
    plt.plot(y_test.iloc[:, idx].values, label=f"Actual {tname}")
    plt.plot(predictions[:, idx],        label=f"Predicted {tname}")
    plt.legend()
    plt.title(f"{tname} Prediction")
    plt.xlabel("Samples")
    plt.ylabel(tname)
    plt.tight_layout()
    img_path = documents_dir / f"{tname}_prediction.png"
    plt.savefig(img_path)
    image_paths.append(img_path)
    plt.close()


# ─────────────────────────────────────────────
# Report Generation & Save  (deep NLP version)
# ─────────────────────────────────────────────

import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Extended target statistics ──────────────────────────────────────────
avg_target    = y.mean()
max_target    = y.max()
min_target    = y.min()
std_target    = y.std()
median_target = y.median()
target_label  = ", ".join(target_cols)
n_train, n_test = len(X_train), len(X_test)
n_features      = len(feature_cols)
model_name      = "Multi-output Linear Regression" if len(target_cols) > 1 else "Linear Regression"

# ── 2. Per-target regression metrics ───────────────────────────────────────
per_target_metrics = {}
for i, tname in enumerate(target_cols):
    actual = y_test.iloc[:, i].values
    pred   = predictions[:, i]
    mae    = mean_absolute_error(actual, pred)
    rmse   = np.sqrt(mean_squared_error(actual, pred))
    r2     = r2_score(actual, pred)
    mean_a = actual.mean()
    mape   = np.mean(np.abs((actual - pred) / (mean_a + 1e-9))) * 100
    per_target_metrics[tname] = dict(mae=mae, rmse=rmse, r2=r2, mape=mape,
                                     mean_actual=mean_a)

# ── 3. Feature correlation with each target ────────────────────────────────
feature_corr = {}
for tname in target_cols:
    corr_series = X.corrwith(y[tname]).sort_values(key=abs, ascending=False)
    feature_corr[tname] = corr_series

# ── 4. Extract linear model coefficients (feature importance proxy) ─────────
if len(target_cols) == 1:
    coef_matrix = model.coef_.reshape(1, -1)
else:
    coef_matrix = np.array([est.coef_ for est in model.estimators_])

# ── 5. NLP helpers ──────────────────────────────────────────────────────────

def _r2_quality(r2):
    if r2 >= 0.90: return "excellent"
    if r2 >= 0.75: return "good"
    if r2 >= 0.50: return "moderate"
    if r2 >= 0.25: return "weak"
    return "poor"

def _mape_verdict(mape):
    if mape < 5:   return "highly accurate (< 5 % error)"
    if mape < 10:  return "accurate (< 10 % error)"
    if mape < 20:  return "acceptable (< 20 % error)"
    return "potentially imprecise (≥ 20 % error) – consider feature engineering"

def _corr_strength(c):
    a = abs(c)
    if a >= 0.7: return "strong"
    if a >= 0.4: return "moderate"
    if a >= 0.2: return "weak"
    return "negligible"

def _trend_sentence(tname, metrics):
    r2   = metrics["r2"]
    mape = metrics["mape"]
    qual = _r2_quality(r2)
    verdict = _mape_verdict(mape)
    direction = (
        "The model explains a large share of variance"   if r2 >= 0.75 else
        "The model captures some of the variance"        if r2 >= 0.40 else
        "The model struggles to capture the variance"
    )
    return (
        f"{direction} in '{tname}' (R² = {r2:.3f}, {qual} fit). "
        f"Predictions are {verdict}, with a Mean Absolute Error of {metrics['mae']:.4f} "
        f"and RMSE of {metrics['rmse']:.4f}."
    )

def _coef_paragraph(tname, idx):
    coefs = coef_matrix[idx]
    coef_pairs = sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
    top = coef_pairs[:3]
    lines = []
    for fname, c in top:
        direction = "positively" if c > 0 else "negatively"
        lines.append(f"'{fname}' ({direction} associated, coefficient {c:.4f})")
    return "The top predictors of '{}' by coefficient magnitude are: {}.".format(
        tname, "; ".join(lines)
    )

def _corr_paragraph(tname):
    corr = feature_corr[tname]
    top3 = corr.head(3)
    parts = []
    for fname, c in top3.items():
        parts.append(
            f"'{fname}' ({_corr_strength(c)} {'positive' if c > 0 else 'negative'} "
            f"correlation, r = {c:.3f})"
        )
    return "Pearson correlation analysis shows the strongest linear relationships with '{}': {}.".format(
        tname, "; ".join(parts)
    )

def _data_quality_section():
    lines = []
    missing_total = data.isnull().sum().sum()
    if missing_total == 0:
        lines.append("Your data is complete with no missing information.")
    else:
        lines.append(f"There were {missing_total} missing pieces of information, but we handled them.")
    low_var = [c for c in feature_cols if X[c].std() < 1e-6]
    if low_var:
        lines.append(f"Some data columns ({', '.join(low_var)}) don't change much – you might not need them.")
    else:
        lines.append("All your data columns have useful variety.")
    return "\n".join(lines)

def _energy_consumption_analysis():
    analysis = []
    for tname in target_cols:
        avg = avg_target[tname]
        max_val = max_target[tname]
        min_val = min_target[tname]
        median = median_target[tname]
        std = std_target[tname]
        analysis.append(f"• {tname} Energy Use:")
        analysis.append(f"  - Average daily consumption: {avg:.2f} units")
        analysis.append(f"  - Highest recorded: {max_val:.2f} units")
        analysis.append(f"  - Lowest recorded: {min_val:.2f} units")
        analysis.append(f"  - Typical (median) use: {median:.2f} units")
        analysis.append(f"  - Variability: {std:.2f} units standard deviation")
        if len(data) > 1:
            first_half = data[tname][:len(data)//2].mean()
            second_half = data[tname][len(data)//2:].mean()
            trend = "increasing" if second_half > first_half else "decreasing" if second_half < first_half else "stable"
            change_pct = abs((second_half - first_half) / first_half * 100) if first_half != 0 else 0
            analysis.append(f"  - Trend: Energy use is {trend} ({change_pct:.1f}% change over time)")
        analysis.append(f"  - Total energy used in period: {data[tname].sum():.2f} units")
        analysis.append(f"  - Days analyzed: {len(data)}")
    return "\n".join(analysis)

def _energy_saving_suggestions():
    suggestions = []
    for i, tname in enumerate(target_cols):
        suggestions.append(f"• To save energy on {tname}, try reducing related activities or factors.")
    suggestions.append("• Switch to LED bulbs to use less electricity.")
    suggestions.append("• Turn off lights and appliances when not in use.")
    suggestions.append("• Use energy-efficient settings on your heating and cooling systems.")
    suggestions.append("• Keep your equipment well-maintained to avoid wasting energy.")
    return "\n".join(suggestions)

def _energy_summary():
    summary = []
    summary.append("This report analyzes your energy consumption patterns based on the provided data.")
    summary.append(f"Data covers {len(data)} records from {DATA_FILE}.")
    summary.append(f"Key metrics tracked: {', '.join(target_cols)}")
    summary.append(f"Factors considered: {', '.join(feature_cols)}")
    summary.append("The analysis helps identify usage patterns, potential savings, and future trends.")
    return "\n".join(summary)

def _reliability_improvements():
    improvements = []
    improvements.append("To improve energy reliability:")
    improvements.append("• Install backup power systems like generators or batteries for outages.")
    improvements.append("• Use smart meters to monitor usage in real-time and detect issues early.")
    improvements.append("• Regular maintenance of electrical systems to prevent failures.")
    improvements.append("• Diversify energy sources (solar, wind) to reduce dependence on one supply.")
    improvements.append("• Implement load balancing to avoid overloading circuits.")
    return "\n".join(improvements)

def _sustainability_improvements():
    improvements = []
    improvements.append("For better energy sustainability:")
    improvements.append("• Switch to renewable energy sources like solar panels or wind turbines.")
    improvements.append("• Reduce waste by using energy-efficient appliances and LED lighting.")
    improvements.append("• Implement recycling programs for old equipment.")
    improvements.append("• Plant trees and improve insulation to naturally reduce energy needs.")
    improvements.append("• Educate household members on sustainable energy practices.")
    improvements.append("• Track carbon footprint and aim to reduce it over time.")
    return "\n".join(improvements)

def _future_predictions():
    predictions_text = []
    future_features = X.mean() * 1.1
    future_pred = model.predict(future_features.to_frame().T)
    for i, tname in enumerate(target_cols):
        pred_val = future_pred[0][i] if len(target_cols) > 1 else future_pred[0].item()
        predictions_text.append(f"• We predict your {tname} energy use next time might be around {pred_val:.2f} units.")
    predictions_text.append("• This is just an estimate based on current patterns.")
    return "\n".join(predictions_text)

def _recommendations(metrics_dict):
    recs = []
    for tname, m in metrics_dict.items():
        if m["r2"] < 0.50:
            recs.append(
                f"• '{tname}' has a low R² ({m['r2']:.3f}). Consider adding domain-relevant features, "
                "polynomial terms, or trying a non-linear model (e.g. Random Forest, Gradient Boosting)."
            )
        if m["mape"] >= 20:
            recs.append(
                f"• '{tname}' shows high MAPE ({m['mape']:.1f}%). "
                "Outlier removal or target log-transformation may improve accuracy."
            )
        if m["r2"] >= 0.75 and m["mape"] < 10:
            recs.append(
                f"• '{tname}' is performing well. Focus on monitoring data drift over time "
                "and re-training periodically to maintain accuracy."
            )
    if n_train < 100:
        recs.append(
            f"• The training set is small ({n_train} rows). Gathering more data will likely "
            "improve generalisation and reduce overfitting risk."
        )
    if n_features >= 10:
        recs.append(
            f"• With {n_features} features, consider running a feature-importance or SHAP analysis "
            "to prune low-value predictors and reduce model complexity."
        )
    recs.append("• Apply cross-validation (e.g., 5-fold) for a more robust performance estimate.")
    recs.append("• Store model artifacts (pickle / joblib) and version them alongside this report.")
    return "\n".join(recs) if recs else "• Model performance looks solid. Continue monitoring in production."

# ── 6. Build the deep report ────────────────────────────────────────────────

section_sep  = "─" * 60
thin_sep     = "· " * 30

per_target_blocks = []
for i, tname in enumerate(target_cols):
    m = per_target_metrics[tname]
    block = f"""
  Target: {tname}
  {thin_sep}
  Descriptive Statistics
    Mean   : {avg_target[tname]:.4f}   |  Median : {median_target[tname]:.4f}
    Std Dev: {std_target[tname]:.4f}   |  Range  : [{min_target[tname]:.4f}, {max_target[tname]:.4f}]

  Model Performance
    R²   (coefficient of determination) : {m['r2']:.4f}   → {_r2_quality(m['r2'])} fit
    MAE  (mean absolute error)          : {m['mae']:.4f}
    RMSE (root mean squared error)      : {m['rmse']:.4f}
    MAPE (mean abs percentage error)    : {m['mape']:.2f} %

  Interpretation
    {_trend_sentence(tname, m)}

  Feature Influence (coefficient-based)
    {_coef_paragraph(tname, i)}

  Correlation Insights
    {_corr_paragraph(tname)}
"""
    per_target_blocks.append(block)

report = f"""
{'=' * 60}
  ENERGY REPORT
{'=' * 60}
Generated on: {datetime.now().strftime('%Y-%m-%d at %H:%M')}
Data file: {DATA_FILE}
Total records: {len(X)}

{section_sep}
ENERGY DATA SUMMARY
{section_sep}
{_energy_summary()}

{section_sep}
DETAILED ENERGY CONSUMPTION ANALYSIS
{section_sep}
{_energy_consumption_analysis()}

{section_sep}
ENERGY SAVING TIPS
{section_sep}
{_energy_saving_suggestions()}

{section_sep}
FUTURE ENERGY PREDICTIONS
{section_sep}
{_future_predictions()}

{section_sep}
RELIABILITY IMPROVEMENTS
{section_sep}
{_reliability_improvements()}

{section_sep}
SUSTAINABILITY IMPROVEMENTS
{section_sep}
{_sustainability_improvements()}

{section_sep}
DATA SUMMARY
{section_sep}
{_data_quality_section()}

{'=' * 60}
END OF ENERGY REPORT
{'=' * 60}
"""

# Save NLP report text to in.txt
text_report_path = documents_dir / "in.txt"
with open(text_report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"\n  NLP text report saved to:\n   {text_report_path}\n")

# Open File Explorer with the generated report file selected on Windows
if sys.platform == "win32":
    try:
        import subprocess
        subprocess.Popen(f'explorer /select,"{text_report_path}"')
    except Exception as e:
        print(f"Could not open File Explorer: {e}")
else:
    print("File Explorer opening is only supported on Windows in this script.")