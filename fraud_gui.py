import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import joblib
import re
import csv
from datetime import datetime

MODEL_PATH = 'fraud_nlp_model.pkl'

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Load model ────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)

# ── Main window ───────────────────────────────────────────────────
root = tk.Tk()
root.title("Fraud Detector")
root.geometry("700x560")
root.configure(bg="#f5f5f5")

# ── Notebook (tabs) ───────────────────────────────────────────────
from tkinter import ttk
nb = ttk.Notebook(root)
nb.pack(fill='both', expand=True, padx=16, pady=12)

tab1 = tk.Frame(nb, bg="#f5f5f5")
tab2 = tk.Frame(nb, bg="#f5f5f5")
tab3 = tk.Frame(nb, bg="#f5f5f5")
nb.add(tab1, text="  Single Check  ")
nb.add(tab2, text="  Batch Scan  ")
nb.add(tab3, text="  History  ")

history = []

# ════════════════════════════════════════════════
# TAB 1 — Single check
# ════════════════════════════════════════════════
tk.Label(tab1, text="Message:", bg="#f5f5f5", font=("Helvetica", 10)).pack(anchor='w', padx=12, pady=(12,2))

input_box = scrolledtext.ScrolledText(tab1, height=6, font=("Helvetica", 11), wrap=tk.WORD)
input_box.pack(fill='x', padx=12)

result_var = tk.StringVar(value="")
result_lbl = tk.Label(tab1, textvariable=result_var, font=("Helvetica", 20, "bold"), bg="#f5f5f5")
result_lbl.pack(pady=12)

conf_var = tk.StringVar(value="")
tk.Label(tab1, textvariable=conf_var, font=("Helvetica", 11), bg="#f5f5f5", fg="gray").pack()

def analyze():
    text = input_box.get("1.0", "end-1c").strip()
    if not text:
        messagebox.showwarning("Empty", "Kuch text enter karo.")
        return

    cleaned = clean_text(text)
    pred  = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    is_fraud = pred == 1
    conf = max(proba) * 100

    if is_fraud:
        result_var.set("⚠  FRAUD DETECTED")
        result_lbl.config(fg="#d32f2f")
    else:
        result_var.set("✓  LEGITIMATE")
        result_lbl.config(fg="#2e7d32")

    conf_var.set(f"Confidence: {conf:.1f}%   |   Fraud: {proba[1]*100:.1f}%  Safe: {proba[0]*100:.1f}%")

    ts = datetime.now().strftime("%H:%M:%S")
    history.append(f"[{ts}]  {'FRAUD' if is_fraud else 'SAFE '} ({conf:.1f}%)  {text[:70]}")
    update_history()

def clear_input():
    input_box.delete("1.0", "end")
    result_var.set("")
    conf_var.set("")

btn_row = tk.Frame(tab1, bg="#f5f5f5")
btn_row.pack(pady=10)
tk.Button(btn_row, text="🔍 Analyze", command=analyze,
          bg="#1565c0", fg="white", font=("Helvetica", 11), padx=14).pack(side='left', padx=6)
tk.Button(btn_row, text="✕ Clear",   command=clear_input,
          bg="#757575", fg="white", font=("Helvetica", 11), padx=14).pack(side='left', padx=6)


# ════════════════════════════════════════════════
# TAB 2 — Batch scan
# ════════════════════════════════════════════════
batch_results = []

cols = ("Row", "Label", "Confidence", "Text Preview")
tree = ttk.Treeview(tab2, columns=cols, show='headings', height=16)
for col, w in [("Row",50),("Label",90),("Confidence",100),("Text Preview",420)]:
    tree.heading(col, text=col)
    tree.column(col, width=w)
tree.pack(fill='both', expand=True, padx=12, pady=10)

summary_var = tk.StringVar(value="")
tk.Label(tab2, textvariable=summary_var, bg="#f5f5f5", fg="#555").pack()

def open_csv():
    path = filedialog.askopenfilename(filetypes=[("CSV/XLS", "*.csv *.xls *.txt")])
    if not path:
        return
    for row in tree.get_children():
        tree.delete(row)
    batch_results.clear()

    with open(path, encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fraud_count = 0
    for i, row in enumerate(rows, 1):
        keys = list(row.keys())
        text_key = next((k for k in keys if k.strip().lower() == 'text'), keys[0])
        text = row[text_key]
        cleaned = clean_text(text)
        pred  = model.predict([cleaned])[0]
        proba = model.predict_proba([cleaned])[0]
        is_fraud = pred == 1
        if is_fraud:
            fraud_count += 1
        label = "⚠ FRAUD" if is_fraud else "✓ SAFE"
        conf  = f"{max(proba)*100:.1f}%"
        tree.insert('', 'end', values=(i, label, conf, text[:80]))
        batch_results.append({"row": i, "label": label, "confidence": conf, "text": text})

    summary_var.set(f"Total: {len(rows)}  |  Fraud: {fraud_count}  |  Safe: {len(rows)-fraud_count}")

def export_csv():
    if not batch_results:
        messagebox.showinfo("Empty", "Pehle CSV scan karo.")
        return
    path = filedialog.asksaveasfilename(defaultextension=".csv")
    if not path:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=["row","label","confidence","text"])
        w.writeheader()
        w.writerows(batch_results)
    messagebox.showinfo("Saved", f"Results saved:\n{path}")

btn_row2 = tk.Frame(tab2, bg="#f5f5f5")
btn_row2.pack(pady=6)
tk.Button(btn_row2, text="📂 Open CSV", command=open_csv,
          bg="#e65100", fg="white", font=("Helvetica", 10), padx=12).pack(side='left', padx=6)
tk.Button(btn_row2, text="💾 Export Results", command=export_csv,
          bg="#2e7d32", fg="white", font=("Helvetica", 10), padx=12).pack(side='left', padx=6)


# ════════════════════════════════════════════════
# TAB 3 — History
# ════════════════════════════════════════════════
hist_box = scrolledtext.ScrolledText(tab3, font=("Courier", 10), state='disabled',
                                      bg="#1e1e1e", fg="#d4d4d4", wrap=tk.WORD)
hist_box.pack(fill='both', expand=True, padx=12, pady=10)

def update_history():
    hist_box.config(state='normal')
    hist_box.delete("1.0", "end")
    for line in reversed(history):
        hist_box.insert("end", line + "\n")
    hist_box.config(state='disabled')

tk.Button(tab3, text="🗑 Clear History",
          command=lambda: [history.clear(), update_history()],
          bg="#757575", fg="white", font=("Helvetica", 10)).pack(pady=6)

# ── Run ───────────────────────────────────────────────────────────
root.mainloop()