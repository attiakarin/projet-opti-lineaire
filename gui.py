"""\ngui.py\n------\nInterface graphique (Tkinter) pour saisir un problème de PL\net afficher les résultats du simplexe (tableaux + solution).\n"""

import io
import contextlib
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog

from simplexe import (
    build_initial_tableau, solve_steps, extract_solution,
    print_tableau, choose_entering_variable,
    choose_leaving_variable, perform_pivot,
)
from two_phase import solve_two_phase
from primal_dual import solve_primal_dual
from branch_and_bound import solve_branch_and_bound


# ---------------------------------------------------------------------------
# Fenêtre principale
# ---------------------------------------------------------------------------

# Couleurs utilisées pour la mise en évidence dans la zone de texte
_TAG_TITLE   = "tag_title"    # titres de section
_TAG_OPTIMAL = "tag_optimal"  # ligne de résultat optimal
_TAG_PIVOT   = "tag_pivot"    # ligne de pivot


class SimplexeApp(tk.Tk):
    """Fenêtre principale du solveur simplexe."""

    def __init__(self):
        super().__init__()
        self.title("Solveur Simplexe")
        self.minsize(860, 620)
        self.configure(bg="#f0f0f0")

        # État du mode pas à pas
        self._st           = None   # SimplexTableau courant (simplexe direct)
        self._iteration    = 0      # numéro d'itération courant
        self._step_done    = False  # True quand fini (optimal / non borné)
        self._trace_buf    = io.StringIO()  # accumule tout le texte pour export
        self._step_chunks  = None   # liste de chunks pré-calculés (autres méthodes)
        self._step_index   = 0      # indice du prochain chunk à afficher

        self._build_ui()

    # ------------------------------------------------------------------
    # Construction de l'interface
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Construit l'ensemble de l'interface."""
        # --- Conteneur principal (deux colonnes) -----------------------
        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Colonne gauche : saisie
        left = ttk.Frame(main, padding=6)
        main.add(left, weight=1)
        self._build_input_panel(left)

        # Colonne droite : sortie
        right = ttk.Frame(main, padding=6)
        main.add(right, weight=3)
        self._build_output_panel(right)

    def _build_input_panel(self, parent):
        """Panneau gauche : champs de saisie + boutons."""
        ttk.Label(parent, text="Solveur Simplexe",
                  font=("Helvetica", 13, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 10))

        # ── c ──────────────────────────────────────────────────────────
        ttk.Label(parent, text="Objectif  c").grid(
            row=1, column=0, sticky="w", pady=3)
        ttk.Label(parent,
                  text="(coefficients séparés par des virgules)",
                  foreground="gray").grid(row=2, column=0, sticky="w")
        self.entry_c = ttk.Entry(parent, width=28)
        self.entry_c.grid(row=1, column=1, rowspan=1, sticky="ew", padx=(6, 0))
        self.entry_c.insert(0, "5, 4")

        ttk.Separator(parent, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=6)

        # ── A ──────────────────────────────────────────────────────────
        ttk.Label(parent, text="Matrice  A").grid(
            row=4, column=0, sticky="nw", pady=3)
        ttk.Label(parent,
                  text="(une contrainte par ligne)",
                  foreground="gray").grid(row=5, column=0, sticky="w")
        self.text_A = tk.Text(parent, width=28, height=6,
                              font=("Courier", 10), relief="solid", bd=1)
        self.text_A.grid(row=4, column=1, rowspan=3, sticky="ew", padx=(6, 0))
        self.text_A.insert("1.0", "6, 4\n1, 2\n-1, 1")

        ttk.Separator(parent, orient="horizontal").grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=6)

        # ── b ──────────────────────────────────────────────────────────
        ttk.Label(parent, text="Second membre  b").grid(
            row=8, column=0, sticky="w", pady=3)
        ttk.Label(parent,
                  text="(valeurs séparées par des virgules)",
                  foreground="gray").grid(row=9, column=0, sticky="w")
        self.entry_b = ttk.Entry(parent, width=28)
        self.entry_b.grid(row=8, column=1, sticky="ew", padx=(6, 0))
        self.entry_b.insert(0, "24, 6, 1")

        ttk.Separator(parent, orient="horizontal").grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=6)

        # ── Choix de méthode ───────────────────────────────────────────
        ttk.Label(parent, text="Méthode :").grid(
            row=11, column=0, sticky="w")
        self.method_var = tk.StringVar(value="simplexe")
        ttk.Radiobutton(parent, text="Simplexe (origine admissible)",
                        variable=self.method_var,
                        value="simplexe").grid(
            row=11, column=1, sticky="w", padx=(6, 0))
        ttk.Radiobutton(parent, text="Deux phases (origine quelconque)",
                        variable=self.method_var,
                        value="deux_phases").grid(
            row=12, column=1, sticky="w", padx=(6, 0))
        ttk.Radiobutton(parent, text="Primal-dual (analyse duale)",
                        variable=self.method_var,
                        value="primal_dual").grid(
            row=13, column=1, sticky="w", padx=(6, 0))
        ttk.Radiobutton(parent, text="Branch & Bound (PLNE entier)",
                        variable=self.method_var,
                        value="branch_bound").grid(
            row=14, column=1, sticky="w", padx=(6, 0))

        ttk.Separator(parent, orient="horizontal").grid(
            row=15, column=0, columnspan=2, sticky="ew", pady=10)

        # ── Boutons ────────────────────────────────────────────────────
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=16, column=0, columnspan=2)

        ttk.Button(btn_frame, text="▶  Résoudre",
                   command=self._on_solve).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="⏭  Pas à pas",
                   command=self._on_step_init).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="✕  Réinitialiser",
                   command=self._on_reset).pack(side=tk.LEFT, padx=4)

        # Deuxième rangée : navigation pas à pas
        btn_frame2 = ttk.Frame(parent)
        btn_frame2.grid(row=17, column=0, columnspan=2, pady=(4, 0))

        self.btn_next = ttk.Button(btn_frame2, text="▶  Suivant",
                                   command=self._on_step_next,
                                   state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=4)

        self.btn_export = ttk.Button(btn_frame2, text="💾  Exporter traces",
                                     command=self._on_export,
                                     state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=4)

        self.lbl_iter = ttk.Label(btn_frame2, text="", foreground="#555")
        self.lbl_iter.pack(side=tk.LEFT, padx=8)

        parent.columnconfigure(1, weight=1)

    def _build_output_panel(self, parent):
        """Panneau droit : zone de texte scrollable avec mise en couleur."""
        ttk.Label(parent, text="Tableaux du simplexe",
                  font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(0, 4))

        self.output_text = scrolledtext.ScrolledText(
            parent,
            font=("Courier", 10),
            state="disabled",
            wrap=tk.NONE,
            relief="solid",
            bd=1,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="white",
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Scrollbar horizontale
        h_scroll = ttk.Scrollbar(parent, orient=tk.HORIZONTAL,
                                 command=self.output_text.xview)
        h_scroll.pack(fill=tk.X)
        self.output_text.configure(xscrollcommand=h_scroll.set)

        # Tags de couleur
        self.output_text.tag_configure(_TAG_TITLE,
                                       foreground="#569cd6",
                                       font=("Courier", 10, "bold"))
        self.output_text.tag_configure(_TAG_OPTIMAL,
                                       foreground="#4ec9b0",
                                       font=("Courier", 10, "bold"))
        self.output_text.tag_configure(_TAG_PIVOT,
                                       foreground="#ce9178")

    # ------------------------------------------------------------------
    # Logique des boutons
    # ------------------------------------------------------------------

    def _on_solve(self):
        """Résout le problème (simplexe ou deux phases) et affiche les tableaux."""
        try:
            c, A, b = self._parse_inputs()
        except ValueError as exc:
            messagebox.showerror("Erreur de saisie", str(exc))
            return

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            if self.method_var.get() == "deux_phases":
                result = solve_two_phase(c, A, b, verbose=True)
            elif self.method_var.get() == "primal_dual":
                result = solve_primal_dual(c, A, b, verbose=True)
            elif self.method_var.get() == "branch_bound":
                result = solve_branch_and_bound(c, A, b, verbose=True)
            else:
                result = solve_steps(c, A, b)

        captured = buffer.getvalue()
        self._trace_buf = io.StringIO()
        self._trace_buf.write(self._trace_header(c, A, b))
        self._trace_buf.write(captured)
        self._render_output(captured, result)
        self.btn_export.config(state=tk.NORMAL)

    def _on_reset(self):
        """Vide les champs, la zone de sortie et l'état pas à pas."""
        self.entry_c.delete(0, tk.END)
        self.text_A.delete("1.0", tk.END)
        self.entry_b.delete(0, tk.END)
        self._set_output("")
        self._st           = None
        self._iteration    = 0
        self._step_done    = False
        self._step_chunks  = None
        self._step_index   = 0
        self._trace_buf    = io.StringIO()
        self.btn_next.config(state=tk.DISABLED)
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_iter.config(text="")

    def _on_step_init(self):
        """Initialise le mode pas à pas et affiche le tableau initial."""
        try:
            c, A, b = self._parse_inputs()
        except ValueError as exc:
            messagebox.showerror("Erreur de saisie", str(exc))
            return

        method = self.method_var.get()
        self._step_done   = False
        self._step_chunks = None
        self._step_index  = 0
        self._trace_buf   = io.StringIO()
        self._trace_buf.write(self._trace_header(c, A, b))

        if method == "simplexe":
            # Mode live : un pivot à la fois
            self._st        = build_initial_tableau(c, A, b)
            self._iteration = 0

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                print("=" * 55)
                print("   MODE PAS À PAS — SIMPLEXE")
                print("=" * 55)
                print_tableau(self._st, title="Tableau initial (itération 0)")

            self._trace_buf.write(buf.getvalue())
            self._set_output(buf.getvalue())
            self.btn_next.config(state=tk.NORMAL)
            self.btn_export.config(state=tk.NORMAL)
            self.lbl_iter.config(text="Itération 0 — appuyez sur Suivant")

        else:
            # Mode pré-calculé : générer tout, découper, afficher pas à pas
            labels = {
                "deux_phases":  "DEUX PHASES",
                "primal_dual":  "PRIMAL-DUAL",
                "branch_bound": "BRANCH & BOUND",
            }
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                if method == "deux_phases":
                    result = solve_two_phase(c, A, b, verbose=True)
                elif method == "primal_dual":
                    result = solve_primal_dual(c, A, b, verbose=True)
                else:
                    result = solve_branch_and_bound(c, A, b, verbose=True)

            full_text = buf.getvalue()
            self._trace_buf.write(full_text)

            self._step_chunks = self._split_into_steps(full_text, method)
            self._step_index  = 0
            self._st          = None

            label = labels.get(method, method)
            header = f"{'='*55}\n   MODE PAS À PAS — {label}\n{'='*55}\n"
            self._set_output(header)

            total = len(self._step_chunks)
            if total == 0:
                self._step_done = True
                self.btn_next.config(state=tk.DISABLED)
                self.lbl_iter.config(text="Aucune étape générée")
            else:
                # Affiche le premier chunk (en-tête de la méthode)
                self._append_output(self._step_chunks[0])
                self._step_index = 1
                if self._step_index >= total:
                    self._step_done = True
                    self.btn_next.config(state=tk.DISABLED)
                    self.lbl_iter.config(text="✓ Terminé")
                else:
                    self.btn_next.config(state=tk.NORMAL)
                    self.lbl_iter.config(
                        text=f"Étape 1/{total} — appuyez sur Suivant")

            self.btn_export.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Découpage du texte verbose en étapes logiques
    # ------------------------------------------------------------------

    def _split_into_steps(self, text: str, method: str) -> list:
        """
        Découpe le texte verbose d'une méthode en une liste de chunks,
        chaque chunk correspondant à une étape logique (itération / noeud).

        Séparateurs détectés (non indentés) :
          deux_phases  → lignes de '━' (≥ 50 car.)
          primal_dual  → lignes de '-' (≥ 50 car.)
          branch_bound → lignes de '─' (≥ 50 car.)
        """
        if method == "deux_phases":
            sep_char, min_len = "━", 50
        elif method == "primal_dual":
            sep_char, min_len = "-", 50
        else:  # branch_bound
            sep_char, min_len = "─", 50

        chunks  = []
        current = []

        for line in text.splitlines(keepends=True):
            stripped = line.rstrip("\r\n")
            is_sep = (
                not line.startswith((" ", "\t"))
                and stripped.startswith(sep_char)
                and len(stripped) >= min_len
            )
            if is_sep and current:
                chunks.append("".join(current))
                current = [line]
            else:
                current.append(line)

        if current:
            chunks.append("".join(current))

        return [c for c in chunks if c.strip()]

    def _on_step_next(self):
        """Effectue une seule itération et affiche le résultat."""
        if self._step_done:
            return

        # ── Mode pré-calculé (deux_phases / primal_dual / branch_bound) ──
        if self._step_chunks is not None:
            if self._step_index >= len(self._step_chunks):
                self._step_done = True
                self.btn_next.config(state=tk.DISABLED)
                self.lbl_iter.config(text="✓ Terminé")
                return

            chunk = self._step_chunks[self._step_index]
            self._append_output(chunk)
            self.output_text.see(tk.END)
            self._step_index += 1

            total = len(self._step_chunks)
            if self._step_index >= total:
                self._step_done = True
                self.btn_next.config(state=tk.DISABLED)
                self.lbl_iter.config(text="✓ Terminé")
            else:
                self.lbl_iter.config(
                    text=f"Étape {self._step_index}/{total} — appuyez sur Suivant")
            return

        # ── Mode live — simplexe direct ───────────────────────────────────
        if self._st is None:
            return

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print(f"\n{'━'*55}")
            print(f"  Itération {self._iteration}")
            print(f"{'━'*55}")

            # ── 1. Variable entrante ─────────────────────────────────
            entering_col = choose_entering_variable(self._st, verbose=True)

            if entering_col == -1:
                # ── Optimal : affiche tableau sans marqueurs ─────────
                print_tableau(self._st,
                              title=f"Tableau — itération {self._iteration}")
                extract_solution(self._st, verbose=True)
                self._step_done = True
                self.btn_next.config(state=tk.DISABLED)
                self.lbl_iter.config(text="✓ Solution optimale trouvée")
            else:
                entering_name = self._st.var_names[entering_col]

                # ── 2. Variable sortante ─────────────────────────────
                leaving_row = choose_leaving_variable(
                    self._st, entering_col, verbose=True)

                if leaving_row == -1:
                    print("Problème NON BORNÉ — arrêt.")
                    self._step_done = True
                    self.btn_next.config(state=tk.DISABLED)
                    self.lbl_iter.config(text="✗ Problème non borné")
                else:
                    leaving_name  = self._st.var_names[self._st.basis[leaving_row]]
                    pivot_val     = self._st.tableau[leaving_row][entering_col]

                    # ── Tableau avec mise en évidence ────────────────
                    print_tableau(self._st,
                                  title=f"Tableau — itération {self._iteration}",
                                  entering_col=entering_col,
                                  leaving_row=leaving_row)

                    # ── Résumé encadré ───────────────────────────────
                    bar = "┄" * 45
                    print(bar)
                    print(f"  RÉSUMÉ DU PIVOT — itération {self._iteration}")
                    print(f"  Entrante  : {entering_name:>4}  "
                          f"(coeff objectif le plus négatif)")
                    print(f"  Sortante  : {leaving_name:>4}  "
                          f"(ratio minimum)")
                    print(f"  Pivot     : {pivot_val:>8.4f}  "
                          f"(ligne {leaving_row}, colonne {entering_col})")
                    print(bar)

                    # ── 3. Pivot ────────────────────────────────────
                    self._st = perform_pivot(
                        self._st, leaving_row, entering_col, verbose=True)
                    self._iteration += 1
                    self.lbl_iter.config(
                        text=f"Itération {self._iteration} — appuyez sur Suivant")

        self._append_output(buf.getvalue())
        self._trace_buf.write(buf.getvalue())
        self.output_text.see(tk.END)

    # ------------------------------------------------------------------
    # Parsing des entrées
    # ------------------------------------------------------------------

    def _parse_inputs(self):
        """
        Lit et valide c, A, b.
        Lève ValueError avec un message lisible en cas d'erreur.
        """
        raw_c = self.entry_c.get().strip()
        raw_A = self.text_A.get("1.0", tk.END).strip()
        raw_b = self.entry_b.get().strip()

        if not raw_c or not raw_A or not raw_b:
            raise ValueError("Tous les champs doivent être remplis.")

        try:
            c = [float(v) for v in raw_c.split(",")]
        except ValueError:
            raise ValueError("Coefficients objectif invalides.")

        try:
            A = []
            for line in raw_A.splitlines():
                line = line.strip()
                if line:
                    A.append([float(v) for v in line.split(",")])
        except ValueError:
            raise ValueError("Matrice A invalide.")

        try:
            b = [float(v) for v in raw_b.split(",")]
        except ValueError:
            raise ValueError("Second membre b invalide.")

        n = len(c)
        for i, row in enumerate(A):
            if len(row) != n:
                raise ValueError(
                    f"Ligne {i+1} de A : {len(row)} valeurs, attendu {n}.")
        if len(b) != len(A):
            raise ValueError(
                f"b a {len(b)} valeurs mais A a {len(A)} lignes.")

        return c, A, b

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Export des traces
    # ------------------------------------------------------------------

    def _trace_header(self, c, A, b) -> str:
        """Génère l'en-tête du fichier de traces."""
        lines = []
        lines.append("=" * 55)
        lines.append("  SOLVEUR SIMPLEXE — Fichier de traces")
        lines.append(f"  Généré le {datetime.datetime.now():%d/%m/%Y %H:%M:%S}")
        lines.append("=" * 55)
        lines.append(f"  Méthode : {self.method_var.get()}")
        lines.append(f"  c = {c}")
        lines.append("  A :")
        for row in A:
            lines.append(f"      {row}")
        lines.append(f"  b = {b}")
        lines.append("=" * 55 + "\n")
        return "\n".join(lines)

    def _on_export(self):
        """Sauvegarde les traces dans un fichier .txt choisi par l'utilisateur."""
        content = self._trace_buf.getvalue()
        if not content.strip():
            messagebox.showinfo("Export", "Aucune trace à exporter.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")],
            initialfile=f"traces_simplexe_{datetime.datetime.now():%Y%m%d_%H%M%S}.txt",
            title="Enregistrer les traces",
        )
        if not filepath:
            return

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            messagebox.showinfo("Export réussi",
                                f"Traces enregistrées dans :\n{filepath}")
        except OSError as exc:
            messagebox.showerror("Erreur d'écriture", str(exc))

    # ------------------------------------------------------------------
    # Helpers d'affichage
    # ------------------------------------------------------------------

    def _colorize_lines(self, text: str):
        """
        Insère les lignes de `text` dans la zone de sortie
        avec mise en couleur selon leur contenu.
        Ne modifie PAS l'état disabled/normal (le caller s'en charge).
        """
        keywords_title   = ("Tableau", "Itération", "RÉSOLUTION", "MODE PAS",
                             "MÉTHODE", "ETAPE", "Branch", "PLNE", "Noeud",
                             "SÉPARATION", "Phase ")
        keywords_pivot   = ("Pivot", "Entrante", "Sortante", "ratio", "──",
                            "RÉSUMÉ", "┄", "Entrante  :", "Sortante  :",
                            "Pivot     :", "contrainte:", "Z_relax", "Noeuds")
        keywords_optimal = ("✓", "═", "Z  =", "OPTIMAL", "Solution optimale",
                            "NON BORNÉ", "Z*", "entier", "meilleur optimum",
                            "Optimum entier", "elagué")

        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if any(kw in stripped for kw in keywords_optimal):
                tag = _TAG_OPTIMAL
            elif any(kw in stripped for kw in keywords_title):
                tag = _TAG_TITLE
            elif any(kw in stripped for kw in keywords_pivot):
                tag = _TAG_PIVOT
            else:
                tag = ""
            if tag:
                self.output_text.insert(tk.END, line, tag)
            else:
                self.output_text.insert(tk.END, line)

    def _render_output(self, captured: str, result):
        """Remplace le contenu de la zone de sortie (mode résolution complète)."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self._colorize_lines(captured)
        self.output_text.config(state="disabled")
        self.output_text.see("1.0")

    def _append_output(self, text: str):
        """Ajoute du texte coloré à la fin de la zone de sortie (mode pas à pas)."""
        self.output_text.config(state="normal")
        self._colorize_lines(text)
        self.output_text.config(state="disabled")

    def _set_output(self, text: str):
        """Remplace entièrement le contenu de la zone de sortie."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.config(state="disabled")
