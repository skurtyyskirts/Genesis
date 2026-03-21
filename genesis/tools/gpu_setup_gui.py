"""
Genesis GPU Setup & Configuration GUI.

A comprehensive tkinter-based GUI for configuring Genesis physics engine:
- GPU/backend detection and selection
- Scene and solver configuration
- Example browser and launcher
- Dependency checker
"""

import os
import sys
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU / dependency detection (inline to avoid import issues)
# ---------------------------------------------------------------------------

def _detect_devices():
    """Detect available compute backends."""
    devices = []

    # CPU always available
    cpu = {"name": "CPU", "backend": "cpu", "memory_gb": 0, "available": True, "detail": ""}
    try:
        import psutil
        cpu["memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        pass
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        cpu["name"] = info.get("brand_raw", "CPU")
    except Exception:
        pass
    cpu["detail"] = f'{cpu["name"]}  |  {cpu["memory_gb"]} GB RAM'
    devices.append(cpu)

    # CUDA / HIP
    try:
        import torch
        if torch.cuda.is_available():
            is_hip = bool(getattr(torch.version, "hip", None))
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                be = "amdgpu" if is_hip else "cuda"
                drv = torch.version.hip if is_hip else (torch.version.cuda or "")
                mem = round(p.total_memory / 1024**3, 1)
                devices.append({
                    "name": p.name,
                    "backend": be,
                    "memory_gb": mem,
                    "available": True,
                    "detail": f"{p.name}  |  {mem} GB  |  {be.upper()} {drv}",
                })
    except Exception:
        pass

    # Metal
    try:
        import torch
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            devices.append({
                "name": "Apple Metal (MPS)",
                "backend": "metal",
                "memory_gb": cpu["memory_gb"],
                "available": True,
                "detail": f"Apple Metal  |  {cpu['memory_gb']} GB shared",
            })
    except Exception:
        pass

    return devices


def _check_deps():
    """Return list of (name, installed, version) for key packages."""
    import importlib
    pkgs = [
        "torch", "quadrants", "numpy", "trimesh", "mujoco",
        "pyrender", "pyglet", "PIL", "pydantic", "cv2",
    ]
    results = []
    for pkg in pkgs:
        try:
            m = importlib.import_module(pkg)
            results.append((pkg, True, getattr(m, "__version__", "ok")))
        except ImportError:
            results.append((pkg, False, "-"))
    return results


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENESIS_DIR = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _GENESIS_DIR.parent / "examples"
_ASSETS_DIR = _GENESIS_DIR / "assets"

BACKEND_CHOICES = ["auto", "cpu", "cuda", "amdgpu", "metal"]
PRECISION_CHOICES = ["32", "64"]
THEME_CHOICES = ["dark", "light", "dumb"]
INTEGRATOR_CHOICES = ["Euler", "implicitfast", "approximate_implicitfast"]
CONSTRAINT_SOLVER_CHOICES = ["CG", "Newton"]

# Example categories
EXAMPLE_CATEGORIES = {
    "Tutorials": "tutorials",
    "Rigid Body": "rigid",
    "Collision": "collision",
    "Coupling": "coupling",
    "Manipulation": "manipulation",
    "Locomotion": "locomotion",
    "Drone": "drone",
    "Rendering": "rendering",
    "Sensors": "sensors",
    "IPC Solver": "IPC_Solver",
    "SAP Coupling": "sap_coupling",
    "Kinematic": "kinematic",
    "FEM / Deformable": None,  # top-level .py files
    "Speed Benchmark": "speed_benchmark",
    "Viewer Plugin": "viewer_plugin",
}

# Solver types with default option fields
SOLVER_DEFAULTS = {
    "Simulation": {
        "dt": ("0.01", "Time step (seconds)"),
        "substeps": ("1", "Substeps per step"),
        "gravity_z": ("-9.81", "Gravity Z (m/s^2)"),
        "floor_height": ("0.0", "Floor height (m)"),
        "requires_grad": ("False", "Differentiable mode"),
    },
    "Rigid": {
        "enable_collision": ("True", "Enable collision"),
        "enable_joint_limit": ("True", "Enable joint limits"),
        "integrator": ("approximate_implicitfast", "Integrator"),
        "constraint_solver": ("Newton", "Constraint solver"),
        "iterations": ("50", "Solver iterations"),
        "tolerance": ("1e-6", "Solver tolerance"),
        "max_collision_pairs": ("150", "Max collision pairs"),
        "use_contact_island": ("False", "Contact islands"),
        "use_hibernation": ("False", "Hibernation"),
    },
    "MPM": {
        "particle_size": ("auto", "Particle size (m)"),
        "grid_density": ("64", "Grid density (cells/m)"),
        "enable_CPIC": ("False", "Enable CPIC"),
        "lower_bound": ("-1,-1,0", "Domain lower bound"),
        "upper_bound": ("1,1,1", "Domain upper bound"),
    },
    "SPH": {
        "particle_size": ("0.02", "Particle size (m)"),
        "pressure_solver": ("WCSPH", "Pressure solver"),
    },
    "PBD": {
        "particle_size": ("0.01", "Particle size (m)"),
        "iterations": ("10", "Solver iterations"),
    },
    "FEM": {
        "damping": ("0.0", "Damping coefficient"),
    },
}


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------

class GenesisSetupGUI:
    """Main Genesis GPU Setup & Configuration window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Genesis — GPU Setup & Configuration")
        self.root.geometry("960x700")
        self.root.minsize(800, 600)

        # State
        self.devices = []
        self.selected_backend = tk.StringVar(value="auto")
        self.selected_precision = tk.StringVar(value="32")
        self.selected_theme = tk.StringVar(value="dark")
        self.show_viewer = tk.BooleanVar(value=True)
        self.show_fps = tk.BooleanVar(value=False)
        self.debug_mode = tk.BooleanVar(value=False)
        self.performance_mode = tk.BooleanVar(value=False)
        self.seed_var = tk.StringVar(value="")
        self.solver_vars: dict[str, dict[str, tk.StringVar]] = {}
        self._process = None

        self._build_ui()
        self._detect_hardware()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        # Menu bar
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export Config Script...", command=self._export_script)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About Genesis", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_hardware_tab()
        self._build_config_tab()
        self._build_solver_tab()
        self._build_examples_tab()
        self._build_deps_tab()

        # Bottom bar
        bottom = ttk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(bottom, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(bottom, text="Launch Genesis", command=self._launch_genesis).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(bottom, text="Generate Script", command=self._export_script).pack(side=tk.RIGHT)

    # ---- Tab 1: Hardware --------------------------------------------------
    def _build_hardware_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Hardware ")

        # Device list
        lf = ttk.LabelFrame(tab, text="Detected Devices")
        lf.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        cols = ("backend", "device", "memory", "status")
        self.dev_tree = ttk.Treeview(lf, columns=cols, show="headings", height=6)
        self.dev_tree.heading("backend", text="Backend")
        self.dev_tree.heading("device", text="Device")
        self.dev_tree.heading("memory", text="Memory (GB)")
        self.dev_tree.heading("status", text="Status")
        self.dev_tree.column("backend", width=80)
        self.dev_tree.column("device", width=350)
        self.dev_tree.column("memory", width=100)
        self.dev_tree.column("status", width=100)
        self.dev_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        btn_frame = ttk.Frame(lf)
        btn_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(btn_frame, text="Refresh", command=self._detect_hardware).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Run Benchmark", command=self._run_benchmark).pack(side=tk.LEFT, padx=4)

        # Selection
        sel_frame = ttk.LabelFrame(tab, text="Backend Selection")
        sel_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        row = ttk.Frame(sel_frame)
        row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row, text="Backend:").pack(side=tk.LEFT)
        be_combo = ttk.Combobox(row, textvariable=self.selected_backend, values=BACKEND_CHOICES, width=12, state="readonly")
        be_combo.pack(side=tk.LEFT, padx=6)

        ttk.Label(row, text="Precision:").pack(side=tk.LEFT, padx=(16, 0))
        pr_combo = ttk.Combobox(row, textvariable=self.selected_precision, values=PRECISION_CHOICES, width=6, state="readonly")
        pr_combo.pack(side=tk.LEFT, padx=6)

        self.recommendation_var = tk.StringVar(value="")
        ttk.Label(sel_frame, textvariable=self.recommendation_var, foreground="green").pack(padx=8, pady=(0, 6), anchor=tk.W)

    # ---- Tab 2: Configuration ---------------------------------------------
    def _build_config_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Configuration ")

        # Genesis init options
        lf = ttk.LabelFrame(tab, text="Genesis Init Options")
        lf.pack(fill=tk.X, padx=8, pady=8)

        grid = ttk.Frame(lf)
        grid.pack(fill=tk.X, padx=8, pady=6)

        r = 0
        ttk.Label(grid, text="Theme:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(grid, textvariable=self.selected_theme, values=THEME_CHOICES, width=10, state="readonly").grid(
            row=r, column=1, sticky=tk.W, padx=4, pady=2
        )

        r += 1
        ttk.Label(grid, text="Random Seed:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(grid, textvariable=self.seed_var, width=12).grid(row=r, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Label(grid, text="(leave empty for none)").grid(row=r, column=2, sticky=tk.W, padx=4, pady=2)

        r += 1
        ttk.Checkbutton(grid, text="Debug mode (CPU only, extra checks)", variable=self.debug_mode).grid(
            row=r, column=0, columnspan=3, sticky=tk.W, pady=2
        )
        r += 1
        ttk.Checkbutton(grid, text="Performance mode (disable ndarray/fastcache)", variable=self.performance_mode).grid(
            row=r, column=0, columnspan=3, sticky=tk.W, pady=2
        )

        # Scene options
        lf2 = ttk.LabelFrame(tab, text="Scene Options")
        lf2.pack(fill=tk.X, padx=8, pady=(0, 8))

        grid2 = ttk.Frame(lf2)
        grid2.pack(fill=tk.X, padx=8, pady=6)

        ttk.Checkbutton(grid2, text="Show interactive viewer", variable=self.show_viewer).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=2
        )
        ttk.Checkbutton(grid2, text="Show FPS", variable=self.show_fps).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=2
        )

        # Viewer options
        lf3 = ttk.LabelFrame(tab, text="Viewer Options")
        lf3.pack(fill=tk.X, padx=8, pady=(0, 8))

        grid3 = ttk.Frame(lf3)
        grid3.pack(fill=tk.X, padx=8, pady=6)

        self.viewer_res_w = tk.StringVar(value="1280")
        self.viewer_res_h = tk.StringVar(value="720")
        self.viewer_max_fps = tk.StringVar(value="60")
        self.cam_fov = tk.StringVar(value="40")

        r = 0
        ttk.Label(grid3, text="Resolution:").grid(row=r, column=0, sticky=tk.W, pady=2)
        res_frame = ttk.Frame(grid3)
        res_frame.grid(row=r, column=1, sticky=tk.W, padx=4, pady=2)
        ttk.Entry(res_frame, textvariable=self.viewer_res_w, width=6).pack(side=tk.LEFT)
        ttk.Label(res_frame, text=" x ").pack(side=tk.LEFT)
        ttk.Entry(res_frame, textvariable=self.viewer_res_h, width=6).pack(side=tk.LEFT)

        r += 1
        ttk.Label(grid3, text="Max FPS:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(grid3, textvariable=self.viewer_max_fps, width=8).grid(row=r, column=1, sticky=tk.W, padx=4, pady=2)

        r += 1
        ttk.Label(grid3, text="Camera FOV:").grid(row=r, column=0, sticky=tk.W, pady=2)
        ttk.Entry(grid3, textvariable=self.cam_fov, width=8).grid(row=r, column=1, sticky=tk.W, padx=4, pady=2)

        # Environment variables
        lf4 = ttk.LabelFrame(tab, text="Environment Variables")
        lf4.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        self.env_text = tk.Text(lf4, height=5, width=60, font=("Consolas", 10))
        self.env_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self.env_text.insert("1.0", "# Set environment variables (KEY=VALUE per line)\n# GS_ENABLE_ZEROCOPY=1\n# QD_NUM_THREADS=1\n")

    # ---- Tab 3: Solver Options --------------------------------------------
    def _build_solver_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Solvers ")

        # Create a notebook within the tab for solver sub-tabs
        solver_nb = ttk.Notebook(tab)
        solver_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        for solver_name, fields in SOLVER_DEFAULTS.items():
            stab = ttk.Frame(solver_nb)
            solver_nb.add(stab, text=f" {solver_name} ")

            self.solver_vars[solver_name] = {}

            # Scrollable frame
            canvas = tk.Canvas(stab, borderwidth=0, highlightthickness=0)
            scrollbar = ttk.Scrollbar(stab, orient=tk.VERTICAL, command=canvas.yview)
            frame = ttk.Frame(canvas)

            canvas.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            canvas.create_window((0, 0), window=frame, anchor="nw")
            frame.bind("<Configure>", lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")))

            for r, (key, (default, description)) in enumerate(fields.items()):
                ttk.Label(frame, text=f"{description}:", width=24, anchor=tk.W).grid(
                    row=r, column=0, sticky=tk.W, padx=(8, 4), pady=3
                )
                var = tk.StringVar(value=default)
                self.solver_vars[solver_name][key] = var

                # Boolean fields get a combobox
                if default in ("True", "False"):
                    w = ttk.Combobox(frame, textvariable=var, values=["True", "False"], width=14, state="readonly")
                elif key == "integrator":
                    w = ttk.Combobox(frame, textvariable=var, values=INTEGRATOR_CHOICES, width=28, state="readonly")
                elif key == "constraint_solver":
                    w = ttk.Combobox(frame, textvariable=var, values=CONSTRAINT_SOLVER_CHOICES, width=14, state="readonly")
                elif key == "pressure_solver":
                    w = ttk.Combobox(frame, textvariable=var, values=["WCSPH", "DFSPH"], width=14, state="readonly")
                else:
                    w = ttk.Entry(frame, textvariable=var, width=16)

                w.grid(row=r, column=1, sticky=tk.W, padx=4, pady=3)
                ttk.Label(frame, text=key, foreground="gray").grid(row=r, column=2, sticky=tk.W, padx=4, pady=3)

    # ---- Tab 4: Examples --------------------------------------------------
    def _build_examples_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Examples ")

        paned = ttk.PanedWindow(tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: category + file list
        left = ttk.Frame(paned)
        paned.add(left, weight=1)

        ttk.Label(left, text="Category:").pack(anchor=tk.W, padx=4, pady=(4, 0))
        self.cat_var = tk.StringVar(value="Tutorials")
        cat_combo = ttk.Combobox(left, textvariable=self.cat_var, values=list(EXAMPLE_CATEGORIES.keys()), state="readonly")
        cat_combo.pack(fill=tk.X, padx=4, pady=4)
        cat_combo.bind("<<ComboboxSelected>>", self._on_category_change)

        self.file_listbox = tk.Listbox(left, font=("Consolas", 10))
        self.file_listbox.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self.file_listbox.bind("<<ListboxSelect>>", self._on_example_select)

        btn_row = ttk.Frame(left)
        btn_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(btn_row, text="Run Selected", command=self._run_example).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Open in Editor", command=self._open_example).pack(side=tk.LEFT, padx=4)

        # Right: preview
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        ttk.Label(right, text="Preview:").pack(anchor=tk.W, padx=4, pady=(4, 0))
        self.preview_text = tk.Text(right, font=("Consolas", 10), wrap=tk.NONE, state=tk.DISABLED)
        preview_scroll_y = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.preview_text.yview)
        preview_scroll_x = ttk.Scrollbar(right, orient=tk.HORIZONTAL, command=self.preview_text.xview)
        self.preview_text.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)
        preview_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        preview_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._example_files: list[Path] = []
        self._on_category_change()

    # ---- Tab 5: Dependencies -----------------------------------------------
    def _build_deps_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text=" Dependencies ")

        cols = ("package", "status", "version")
        self.dep_tree = ttk.Treeview(tab, columns=cols, show="headings", height=12)
        self.dep_tree.heading("package", text="Package")
        self.dep_tree.heading("status", text="Status")
        self.dep_tree.heading("version", text="Version")
        self.dep_tree.column("package", width=200)
        self.dep_tree.column("status", width=120)
        self.dep_tree.column("version", width=200)
        self.dep_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        btn = ttk.Frame(tab)
        btn.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn, text="Refresh", command=self._check_dependencies).pack(side=tk.LEFT)

        self._check_dependencies()

    # ---------------------------------------------------------------- Logic
    def _detect_hardware(self, _event=None):
        self.status_var.set("Detecting hardware...")
        self.root.update_idletasks()

        self.devices = _detect_devices()

        # Populate tree
        for item in self.dev_tree.get_children():
            self.dev_tree.delete(item)

        recommended_be = None
        priority = {"cuda": 0, "amdgpu": 1, "metal": 2, "cpu": 3}
        best_score = 99
        for d in self.devices:
            status = "Available"
            tag = ""
            score = priority.get(d["backend"], 99)
            if score < best_score:
                best_score = score
                recommended_be = d["backend"]
            self.dev_tree.insert("", tk.END, values=(
                d["backend"].upper(), d["name"], f'{d["memory_gb"]:.1f}', status
            ))

        if recommended_be:
            self.recommendation_var.set(f"Recommended: {recommended_be.upper()} backend")
        self.status_var.set(f"Detected {len(self.devices)} device(s)")

    def _check_dependencies(self):
        for item in self.dep_tree.get_children():
            self.dep_tree.delete(item)
        deps = _check_deps()
        for name, installed, version in deps:
            status = "Installed" if installed else "MISSING"
            self.dep_tree.insert("", tk.END, values=(name, status, version))

    def _on_category_change(self, _event=None):
        cat = self.cat_var.get()
        subdir = EXAMPLE_CATEGORIES.get(cat)
        self._example_files = []

        if subdir is not None:
            search_dir = _EXAMPLES_DIR / subdir
            if search_dir.is_dir():
                self._example_files = sorted(search_dir.glob("*.py"))
        else:
            # Top-level example .py files (FEM / Deformable)
            self._example_files = sorted(f for f in _EXAMPLES_DIR.glob("*.py") if f.is_file())

        self.file_listbox.delete(0, tk.END)
        for f in self._example_files:
            self.file_listbox.insert(tk.END, f.name)

    def _on_example_select(self, _event=None):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        fpath = self._example_files[sel[0]]
        try:
            content = fpath.read_text(errors="replace")
        except Exception as e:
            content = f"Error reading file: {e}"

        self.preview_text.configure(state=tk.NORMAL)
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert("1.0", content)
        self.preview_text.configure(state=tk.DISABLED)

    def _run_example(self):
        sel = self.file_listbox.curselection()
        if not sel:
            messagebox.showinfo("No Selection", "Select an example to run.")
            return
        fpath = self._example_files[sel[0]]
        self._run_script(str(fpath))

    def _open_example(self):
        sel = self.file_listbox.curselection()
        if not sel:
            return
        fpath = self._example_files[sel[0]]
        # Try xdg-open / open / start
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(fpath)])
            elif sys.platform == "win32":
                os.startfile(str(fpath))
            else:
                subprocess.Popen(["xdg-open", str(fpath)])
        except Exception:
            messagebox.showinfo("Open File", f"Path: {fpath}")

    def _run_benchmark(self):
        bench_dir = _EXAMPLES_DIR / "speed_benchmark"
        scripts = sorted(bench_dir.glob("*.py")) if bench_dir.is_dir() else []
        if not scripts:
            messagebox.showinfo("Benchmark", "No benchmark scripts found.")
            return
        self._run_script(str(scripts[0]))

    # ---- Script generation ------------------------------------------------
    def _build_init_code(self) -> str:
        """Generate gs.init() call from current settings."""
        be = self.selected_backend.get()
        pr = self.selected_precision.get()
        th = self.selected_theme.get()
        debug = self.debug_mode.get()
        perf = self.performance_mode.get()
        seed = self.seed_var.get().strip()

        parts = []
        if be != "auto":
            parts.append(f"backend=gs.{be}")
        if pr != "32":
            parts.append(f'precision="{pr}"')
        if th != "dark":
            parts.append(f'theme="{th}"')
        if debug:
            parts.append("debug=True")
        if perf:
            parts.append("performance_mode=True")
        if seed:
            parts.append(f"seed={seed}")

        init_args = ", ".join(parts)
        return f"gs.init({init_args})"

    def _build_scene_code(self) -> str:
        """Generate gs.Scene() call from solver config."""
        lines = []

        # SimOptions
        sim_vars = self.solver_vars.get("Simulation", {})
        sim_parts = []
        if sim_vars.get("dt") and sim_vars["dt"].get() != "0.01":
            sim_parts.append(f"dt={sim_vars['dt'].get()}")
        if sim_vars.get("substeps") and sim_vars["substeps"].get() != "1":
            sim_parts.append(f"substeps={sim_vars['substeps'].get()}")
        if sim_vars.get("gravity_z") and sim_vars["gravity_z"].get() != "-9.81":
            gz = sim_vars["gravity_z"].get()
            sim_parts.append(f"gravity=(0.0, 0.0, {gz})")
        if sim_vars.get("floor_height") and sim_vars["floor_height"].get() != "0.0":
            sim_parts.append(f"floor_height={sim_vars['floor_height'].get()}")
        if sim_vars.get("requires_grad") and sim_vars["requires_grad"].get() == "True":
            sim_parts.append("requires_grad=True")
        if sim_parts:
            lines.append(f"sim_options=gs.options.SimOptions({', '.join(sim_parts)})")

        # RigidOptions
        rigid_vars = self.solver_vars.get("Rigid", {})
        rigid_parts = []
        for key, var in rigid_vars.items():
            val = var.get()
            defaults = dict(SOLVER_DEFAULTS["Rigid"])
            if val != defaults.get(key, ("",))[0] if isinstance(defaults.get(key), tuple) else defaults.get(key, ""):
                default_val = SOLVER_DEFAULTS["Rigid"].get(key, ("",))[0]
                if val != default_val:
                    if val in ("True", "False"):
                        rigid_parts.append(f"{key}={val}")
                    elif key == "integrator":
                        rigid_parts.append(f"integrator=gs.integrator.{val}")
                    elif key == "constraint_solver":
                        rigid_parts.append(f"constraint_solver=gs.constraint_solver.{val}")
                    else:
                        rigid_parts.append(f"{key}={val}")
        if rigid_parts:
            lines.append(f"rigid_options=gs.options.RigidOptions({', '.join(rigid_parts)})")

        # Build Scene call
        show_viewer = self.show_viewer.get()
        show_fps = self.show_fps.get()

        scene_args = []
        scene_args.extend(lines)
        if not show_viewer:
            scene_args.append("show_viewer=False")
        if show_fps:
            scene_args.append("show_FPS=True")

        try:
            res_w = int(self.viewer_res_w.get())
            res_h = int(self.viewer_res_h.get())
            if (res_w, res_h) != (1280, 720):
                scene_args.append(f"viewer_options=gs.options.ViewerOptions(res=({res_w}, {res_h}))")
        except ValueError:
            pass

        if scene_args:
            formatted = ",\n    ".join(scene_args)
            return f"scene = gs.Scene(\n    {formatted},\n)"
        return "scene = gs.Scene()"

    def _build_env_lines(self) -> list[str]:
        """Parse env var text."""
        lines = []
        raw = self.env_text.get("1.0", tk.END).strip().split("\n")
        for line in raw:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                lines.append(line)
        return lines

    def _generate_full_script(self) -> str:
        """Generate a complete Genesis setup script."""
        env_lines = self._build_env_lines()

        script = ["import genesis as gs", ""]

        if env_lines:
            script.append("import os")
            for line in env_lines:
                k, v = line.split("=", 1)
                script.append(f'os.environ["{k.strip()}"] = "{v.strip()}"')
            script.append("")

        script.append(self._build_init_code())
        script.append("")
        script.append(self._build_scene_code())
        script.append("")
        script.append("# Add entities here:")
        script.append("# scene.add_entity(gs.morphs.Plane())")
        script.append("# robot = scene.add_entity(gs.morphs.URDF(file='path/to/robot.urdf'))")
        script.append("")
        script.append("scene.build()")
        script.append("")
        script.append("# Simulation loop")
        script.append("for i in range(1000):")
        script.append("    scene.step()")
        script.append("")

        return "\n".join(script)

    def _export_script(self):
        script = self._generate_full_script()
        path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            initialfile="genesis_setup.py",
        )
        if path:
            with open(path, "w") as f:
                f.write(script)
            self.status_var.set(f"Saved: {path}")
            messagebox.showinfo("Saved", f"Script saved to:\n{path}")

    # ---- Launch -----------------------------------------------------------
    def _launch_genesis(self):
        """Launch Genesis with the current configuration."""
        script = self._generate_full_script()

        # Write to temp file and run
        import tempfile

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="genesis_launch_")
        tmp.write(script)
        tmp.flush()
        tmp.close()
        self._run_script(tmp.name)

    def _run_script(self, script_path: str):
        """Run a Python script in a subprocess."""
        if self._process and self._process.poll() is None:
            messagebox.showwarning("Busy", "A script is already running.")
            return

        env = os.environ.copy()
        for line in self._build_env_lines():
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()

        self.status_var.set(f"Running: {os.path.basename(script_path)}")

        def _run():
            try:
                self._process = subprocess.Popen(
                    [sys.executable, script_path],
                    env=env,
                    cwd=str(_GENESIS_DIR.parent),
                )
                self._process.wait()
                self.root.after(0, lambda: self.status_var.set("Script finished"))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Error: {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_about(self):
        try:
            from genesis.version import __version__
            ver = __version__
        except Exception:
            ver = "unknown"
        messagebox.showinfo(
            "About Genesis",
            f"Genesis Physics Engine v{ver}\n\n"
            "A universal and generative physics engine\n"
            "for robotics and embodied AI.\n\n"
            "GPU Setup & Configuration Tool",
        )


def main():
    """Entry point for the Genesis GPU Setup GUI."""
    root = tk.Tk()

    # Try to set a nice theme
    try:
        style = ttk.Style()
        available = style.theme_names()
        for preferred in ("clam", "alt", "default"):
            if preferred in available:
                style.theme_use(preferred)
                break
    except Exception:
        pass

    GenesisSetupGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
