"""
simplexe.py
-----------
Contient toute la logique de l'algorithme du simplexe.

Hypothèses de travail (forme standard) :
  - Maximiser  c^T x
  - Sous contraintes  Ax <= b,  b >= 0,  x >= 0
"""

import numpy as np


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

class ProblemData:
    """Encapsule les données brutes d'un problème de PL."""

    def __init__(self, c, A, b):
        """
        Paramètres
        ----------
        c : list[float]  — coefficients de la fonction objectif (maximisation)
        A : list[list[float]]  — matrice des contraintes (m x n)
        b : list[float]  — second membre des contraintes
        """
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)

    @property
    def num_vars(self):
        return len(self.c)

    @property
    def num_constraints(self):
        return len(self.b)


class SimplexResult:
    """Résultat retourné après la résolution."""

    def __init__(self):
        self.status = "non résolu"   # "optimal", "non borné", "infaisable"
        self.optimal_value = None
        self.solution = None         # valeurs des variables de décision
        self.iterations = []         # liste des tableaux successifs (pour l'affichage)
        self.message = ""


class SimplexTableau:
    """
    Représente le tableau initial du simplexe sous forme de liste de listes.

    Attributs
    ---------
    tableau : list[list[float]]
        (m+1) lignes × (n + m + 1) colonnes.
        Les m premières lignes correspondent aux contraintes ;
        la dernière ligne est la ligne objectif (coefficients de z).
    basis : list[int]
        Indices (dans les colonnes du tableau) des variables basiques,
        une par contrainte.  Initialement ce sont les variables d'écart.
    var_names : list[str]
        Noms des colonnes : ['x1','x2',...,'e1','e2',...,'b'].
    num_vars : int   — nombre de variables de décision (n)
    num_slack : int  — nombre de variables d'écart    (m)
    """

    def __init__(self, tableau, basis, var_names, num_vars, num_slack):
        self.tableau = tableau        # list[list[float]]
        self.basis = basis            # list[int]
        self.var_names = var_names    # list[str]
        self.num_vars = num_vars
        self.num_slack = num_slack

    @property
    def num_constraints(self):
        return len(self.basis)

    def basis_names(self):
        """Retourne les noms des variables basiques actuelles."""
        return [self.var_names[i] for i in self.basis]


# ---------------------------------------------------------------------------
# Construction du tableau initial
# ---------------------------------------------------------------------------

def build_initial_tableau(c, A, b) -> 'SimplexTableau':
    """
    Construit le tableau initial du simplexe **sous forme de listes Python**.

    Paramètres
    ----------
    c : list[float]        — coefficients de la fonction objectif (maximisation)
    A : list[list[float]]  — matrice des contraintes (m × n)
    b : list[float]        — second membre (b >= 0)

    Retourne
    --------
    SimplexTableau
        Tableau augmenté (variables d'écart incluses), base initiale et
        noms de colonnes.

    Structure du tableau (m+1) × (n+m+1) :

        Colonnes : x1 … xn | e1 … em | b
        ┌─────────────────────────────┐
        │   A        I        b      │  ← m lignes contraintes
        ├─────────────────────────────┤
        │  -c        0        0      │  ← ligne objectif  (z row)
        └─────────────────────────────┘

    La base initiale contient les variables d'écart e1 … em
    (indices n, n+1, …, n+m-1 dans le tableau).
    """
    m = len(b)          # nombre de contraintes
    n = len(c)          # nombre de variables de décision
    total_cols = n + m + 1  # variables décision + écart + RHS

    # -- Noms des colonnes ------------------------------------------------
    var_names = [f"x{j+1}" for j in range(n)] + \
                [f"e{i+1}" for i in range(m)] + \
                ["b"]

    # -- Construction ligne par ligne (listes Python) ---------------------
    tableau = []

    for i in range(m):
        row = [0.0] * total_cols
        # coefficients de la contrainte i
        for j in range(n):
            row[j] = float(A[i][j])
        # variable d'écart de la contrainte i → colonne n+i
        row[n + i] = 1.0
        # second membre
        row[-1] = float(b[i])
        tableau.append(row)

    # -- Ligne objectif : on stocke -c (maximisation) ----------------------
    obj_row = [0.0] * total_cols
    for j in range(n):
        obj_row[j] = -float(c[j])
    tableau.append(obj_row)

    # -- Base initiale : variables d'écart e1…em (indices n … n+m-1) ------
    basis = [n + i for i in range(m)]

    return SimplexTableau(tableau, basis, var_names, n, m)


def print_tableau(st: 'SimplexTableau', title: str = "Tableau du simplexe",
                  entering_col: int = -1, leaving_row: int = -1) -> None:
    """
    Affiche le tableau du simplexe au format académique aligné.

    Paramètres optionnels de mise en évidence :
        entering_col : indice de la variable entrante  → marqué « ↑ » dans l'en-tête
        leaving_row  : indice de la ligne sortante     → marqué « → » sur la ligne
        (les deux)   : la cellule pivot est entourée de [ ]
    """
    # Calcul automatique de la largeur des colonnes
    var_names  = st.var_names[:-1]    # sans 'b'
    basis_lbls = st.basis_names()
    all_row_lbls = basis_lbls + ["f"]
    m = st.num_constraints

    # Largeur de la colonne Base = max(len des étiquettes de ligne)
    # +2 pour le préfixe "→ " éventuel
    BASE_W = max(len(l) for l in all_row_lbls) + 1

    # Largeur de chaque colonne numérique : assez large pour les nombres
    # et pour le nom de colonne (au moins 7 caractères)
    all_values = [v for row in st.tableau for v in row]
    max_num_w  = max(len(f"{v:.2f}") for v in all_values)
    COL_W      = max(max_num_w + 1, max((len(n) for n in var_names), default=2) + 1, 7)

    total_inner_w = COL_W * len(var_names)
    sep = "─" * (BASE_W + 3 + total_inner_w + 5 + COL_W)

    def _fmt_cell(value: float, col: int, row: int) -> str:
        """Formate une cellule en ajoutant [  ] si c'est la cellule pivot."""
        num = f"{value:.2f}"
        if entering_col >= 0 and leaving_row >= 0 and col == entering_col and row == leaving_row:
            # Cellule pivot : on entoure de crochets dans la largeur disponible
            bracketed = f"[{num}]"
            return f"{bracketed:>{COL_W}}"
        return f"{value:>{COL_W}.2f}"

    # ── Titre ────────────────────────────────────────────────────────────
    print(f"\n{title}")

    # ── Ligne d'en-tête des noms de variables ────────────────────────────
    header_cells = "".join(
        f"{'↑'+n:>{COL_W}}" if j == entering_col else f"{n:>{COL_W}}"
        for j, n in enumerate(var_names)
    )
    print(f"{'':{BASE_W+2}}{header_cells}  |  {'b':>{COL_W}}")

    # ── Ligne objectif (étiquetée "f") ───────────────────────────────────
    obj   = st.tableau[-1]
    cells = "".join(_fmt_cell(obj[j], j, -1) for j in range(len(var_names)))
    print(f"  {'f':<{BASE_W}} | {cells}  |  {obj[-1]:>{COL_W}.2f}")

    print(sep)

    # ── Lignes des contraintes ───────────────────────────────────────────
    for i in range(m):
        row   = st.tableau[i]
        if i == leaving_row and leaving_row >= 0:
            prefix = "→ "
        else:
            prefix = "  "
        label = basis_lbls[i]
        cells = "".join(_fmt_cell(row[j], j, i) for j in range(len(var_names)))
        print(f"{prefix}{label:<{BASE_W}} | {cells}  |  {row[-1]:>{COL_W}.2f}")

    print()


def choose_entering_variable(st: 'SimplexTableau', verbose: bool = True,
                             forbidden_cols=None) -> int:
    """
    Choisit la variable entrante selon la règle du coefficient le plus négatif
    dans la ligne objectif (règle standard du simplexe).

    Paramètres
    ----------
    st             : SimplexTableau  — tableau courant
    verbose        : bool            — si True, affiche le détail du choix
    forbidden_cols : iterable|None   — colonnes à exclure (ex. variables artificielles)

    Retourne
    --------
    int
        Indice de la colonne choisie (0-indexé dans le tableau).
        Retourne -1 si tous les coefficients sont >= 0 (solution optimale atteinte).

    Principe
    --------
    La ligne objectif est stockée sous la forme  z - c^T x = 0,
    donc les coefficients des variables non basiques sont  -cj.
    Un coefficient **négatif** signifie que faire entrer cette variable
    en base **améliore** la valeur de z.
    On choisit la colonne dont le coefficient est **le plus négatif** :
    c'est la direction qui offre le gain marginal le plus élevé par unité.
    """
    obj_row = st.tableau[-1]          # dernière ligne = ligne z
    var_names = st.var_names          # noms de toutes les colonnes (dont 'b')

    # On ignore la dernière colonne (RHS = b)
    num_cols = len(obj_row) - 1
    excluded = set(forbidden_cols) if forbidden_cols else set()

    if verbose:
        print("─── Choix de la variable entrante ───")
        print("Coefficients de la ligne objectif (z) :")
        for j in range(num_cols):
            if j in excluded:
                continue
            marker = " ← candidat" if obj_row[j] < 0 else ""
            print(f"  {var_names[j]:>4} : {obj_row[j]:>8.4f}{marker}")
        print()

    # Recherche du minimum parmi les colonnes hors RHS
    min_val = 0.0          # seuil : seuls les négatifs nous intéressent
    pivot_col = -1

    for j in range(num_cols):
        if j in excluded:
            continue
        if obj_row[j] < min_val:
            min_val = obj_row[j]
            pivot_col = j

    if verbose:
        if pivot_col == -1:
            print("  ✓ Aucun coefficient négatif → solution optimale, pas de variable entrante.\n")
        else:
            print(f"  → Variable entrante choisie : {var_names[pivot_col]}"
                  f"  (colonne {pivot_col}, coefficient = {min_val:.4f})\n")

    return pivot_col


def choose_leaving_variable(st: 'SimplexTableau', entering_col: int,
                            verbose: bool = True) -> int:
    """
    Choisit la variable sortante par le test du ratio minimum (règle de Bland
    pour le cas de dégénérescence : en cas d'égalité on prend la première ligne).

    Paramètres
    ----------
    st           : SimplexTableau  — tableau courant
    entering_col : int             — indice de la colonne de la variable entrante
    verbose      : bool            — si True, affiche le détail des ratios

    Retourne
    --------
    int
        Indice de la ligne choisie (0-indexé, parmi les m lignes de contraintes).
        Retourne -1 si tous les coefficients de la colonne sont <= 0
        (problème non borné).

    Principe (test du rapport minimum)
    -----------------------------------
    Pour chaque ligne de contrainte i (hors ligne objectif) :
      - Si  a_{i, col} > 0  →  ratio_i = b_i / a_{i, col}
      - Sinon               →  ratio ignoré (la variable peut croître librement
                                sur cette ligne sans violer la contrainte)
    On sélectionne la ligne i* dont le ratio est **minimum** :
      i* = argmin { b_i / a_{i,col}  |  a_{i,col} > 0 }
    Cela garantit que la nouvelle solution de base reste **réalisable** (b >= 0).
    """
    m = st.num_constraints
    var_names = st.var_names
    basis_names = st.basis_names()

    col_coeffs = [st.tableau[i][entering_col] for i in range(m)]
    rhs        = [st.tableau[i][-1]           for i in range(m)]

    if verbose:
        print("─── Choix de la variable sortante (test du ratio) ───")
        print(f"Variable entrante : {var_names[entering_col]}  (colonne {entering_col})")
        print(f"{'Ligne':>5}  {'Base':>5}  {'b':>10}  {'a_col':>10}  {'ratio':>12}")
        print("  " + "─" * 47)

    min_ratio  = None
    leaving_row = -1

    for i in range(m):
        a = col_coeffs[i]
        b_val = rhs[i]

        if a > 1e-9:                      # seuls les coefficients strictement positifs
            ratio = b_val / a
            if verbose:
                print(f"  {i:>3}    {basis_names[i]:>5}  {b_val:>10.4f}  {a:>10.4f}  {ratio:>12.4f}")
            if min_ratio is None or ratio < min_ratio:
                min_ratio   = ratio
                leaving_row = i
        else:
            if verbose:
                marker = "  (ignoré — coefficient ≤ 0)" if a <= 1e-9 else ""
                print(f"  {i:>3}    {basis_names[i]:>5}  {b_val:>10.4f}  {a:>10.4f}  {'—':>12}{marker}")

    if verbose:
        print()
        if leaving_row == -1:
            print("  ✗ Aucun ratio valide → problème NON BORNÉ.\n")
        else:
            print(f"  → Variable sortante : {basis_names[leaving_row]}"
                  f"  (ligne {leaving_row}, ratio = {min_ratio:.4f})\n")

    return leaving_row


def perform_pivot(st: 'SimplexTableau', pivot_row: int, pivot_col: int,
                  verbose: bool = True) -> 'SimplexTableau':
    """
    Effectue une opération de pivot sur le tableau du simplexe.

    Paramètres
    ----------
    st          : SimplexTableau  — tableau courant (non modifié)
    pivot_row   : int             — indice de la ligne pivot (variable sortante)
    pivot_col   : int             — indice de la colonne pivot (variable entrante)
    verbose     : bool            — si True, affiche le pivot et le nouveau tableau

    Retourne
    --------
    SimplexTableau
        Nouveau tableau avec base mise à jour (copie indépendante).

    Algorithme
    ----------
    Étape 1 — Normalisation de la ligne pivot :
        Diviser **toute** la ligne pivot par l'élément pivot, de sorte que
        l'élément pivot devienne exactement 1.

    Étape 2 — Élimination dans les autres lignes :
        Pour chaque ligne i ≠ pivot_row :
            ligne_i  ←  ligne_i  −  (a_{i, col}) × ligne_pivot_normalisée
        Après cette opération la colonne pivot ne contient que des 0,
        sauf sur la ligne pivot où elle vaut 1 → forme canonique.

    Étape 3 — Mise à jour de la base :
        Remplacer la variable sortante (basis[pivot_row]) par l'entrante (pivot_col).
    """
    pivot_val = st.tableau[pivot_row][pivot_col]

    if abs(pivot_val) < 1e-12:
        raise ValueError(
            f"Élément pivot nul ou quasi-nul ({pivot_val:.2e}) en "
            f"({pivot_row}, {pivot_col}) — choix de pivot invalide."
        )

    if verbose:
        entering = st.var_names[pivot_col]
        leaving  = st.var_names[st.basis[pivot_row]]
        print("─── Opération de pivot ───")
        print(f"  Pivot      : tableau[{pivot_row}][{pivot_col}] = {pivot_val:.4f}")
        print(f"  Entrante   : {entering}  (colonne {pivot_col})")
        print(f"  Sortante   : {leaving}   (ligne   {pivot_row})")
        print()

    # -- Copie du tableau (copie superficielle ligne par ligne) ------------
    new_tab  = [row[:] for row in st.tableau]
    new_basis = st.basis[:]
    num_cols  = len(new_tab[0])

    # Étape 1 : normaliser la ligne pivot
    new_tab[pivot_row] = [v / pivot_val for v in new_tab[pivot_row]]

    # Étape 2 : éliminer dans toutes les autres lignes (y compris ligne z)
    total_rows = len(new_tab)
    for i in range(total_rows):
        if i == pivot_row:
            continue
        factor = new_tab[i][pivot_col]
        if abs(factor) < 1e-12:
            continue
        new_tab[i] = [new_tab[i][k] - factor * new_tab[pivot_row][k]
                      for k in range(num_cols)]

    # Étape 3 : mettre à jour la base
    new_basis[pivot_row] = pivot_col

    new_st = SimplexTableau(
        new_tab, new_basis, st.var_names, st.num_vars, st.num_slack
    )

    if verbose:
        print_tableau(new_st, title="Tableau après pivot")

    return new_st


def build_tableau(problem: ProblemData) -> np.ndarray:
    """
    Construit le tableau du simplexe en ajoutant les variables d'écart.

    Disposition du tableau (m+1) x (n + m + 1) :
      [ A | I | b ]
      [-c | 0 | 0 ]   ← ligne de la fonction objectif (z = 0 initialement)
    """
    m, n = problem.num_constraints, problem.num_vars

    tableau = np.zeros((m + 1, n + m + 1))

    # Lignes des contraintes
    tableau[:m, :n] = problem.A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = problem.b

    # Ligne objectif : on stocke -c (car on cherche à maximiser)
    tableau[m, :n] = -problem.c

    return tableau


# ---------------------------------------------------------------------------
# Opérations de pivot
# ---------------------------------------------------------------------------

def find_pivot_column(tableau: np.ndarray) -> int:
    """
    Règle de Bland simplifiée : choisit la colonne dont le coeff
    dans la ligne objectif est le plus négatif.
    Retourne -1 si toutes les valeurs sont >= 0 (solution optimale).
    """
    obj_row = tableau[-1, :-1]
    col = int(np.argmin(obj_row))
    if obj_row[col] >= 0:
        return -1
    return col


def find_pivot_row(tableau: np.ndarray, col: int) -> int:
    """
    Test du rapport minimum pour déterminer la ligne pivotale.
    Retourne -1 si le problème est non borné.
    """
    m = tableau.shape[0] - 1
    rhs = tableau[:m, -1]
    col_vals = tableau[:m, col]

    ratios = np.full(m, np.inf)
    for i in range(m):
        if col_vals[i] > 1e-9:
            ratios[i] = rhs[i] / col_vals[i]

    if np.all(ratios == np.inf):
        return -1

    return int(np.argmin(ratios))


def pivot(tableau: np.ndarray, row: int, col: int) -> np.ndarray:
    """Effectue une opération de pivot sur le tableau."""
    t = tableau.copy()
    pivot_val = t[row, col]
    t[row] /= pivot_val

    for i in range(t.shape[0]):
        if i != row:
            t[i] -= t[i, col] * t[row]

    return t


# ---------------------------------------------------------------------------
# Lecture de la solution optimale depuis le tableau final
# ---------------------------------------------------------------------------

def extract_solution(st: 'SimplexTableau', verbose: bool = True) -> dict:
    """
    Lit la solution optimale depuis le tableau final du simplexe.

    Paramètres
    ----------
    st      : SimplexTableau  — tableau à l'état final (tous coefficients z >= 0)
    verbose : bool            — si True, affiche la solution de façon lisible

    Retourne
    --------
    dict avec les clés :
        'status'         : str   — "Solution optimale trouvée"
        'Z'              : float — valeur optimale de la fonction objectif
        'variables'      : dict  — { 'x1': val, 'x2': val, … }  (vars de décision)

    Principe d'extraction
    ----------------------
    Une variable de décision xj est **basique** si et seulement si son indice j
    figure dans st.basis.  Dans ce cas sa valeur est le RHS de la ligne
    correspondante : x_j = st.tableau[row][-1].
    Toute variable de décision **non basique** est nulle (= 0).
    La valeur de Z se lit directement dans la colonne b de la ligne objectif :
    Z = st.tableau[-1][-1].
    """
    n = st.num_vars

    # Valeurs des variables de décision — lookup O(1) via dict
    basis_map = {col: row for row, col in enumerate(st.basis)}
    variables = {}
    for j in range(n):
        row_idx = basis_map.get(j)
        variables[st.var_names[j]] = (
            st.tableau[row_idx][-1] if row_idx is not None else 0.0
        )

    Z = st.tableau[-1][-1]

    if verbose:
        bar = "═" * 45
        print(bar)
        print("  ✓  Solution optimale trouvée")
        print(bar)
        for name, val in variables.items():
            print(f"  {name}  =  {val:.4f}")
        print(f"  {'─'*20}")
        print(f"  Z  =  {Z:.4f}")
        print(bar)
        print()

    return {
        "status":    "Solution optimale trouvée",
        "Z":         Z,
        "variables": variables,
    }


# ---------------------------------------------------------------------------
# Boucle complète du simplexe (pas à pas, sur SimplexTableau)
# ---------------------------------------------------------------------------

def solve_steps(c, A, b, max_iter: int = 100) -> SimplexResult:
    """
    Résout un problème de PL par la méthode du simplexe en affichant
    chaque étape : tableau, variable entrante, variable sortante, pivot.

    Paramètres
    ----------
    c, A, b   : listes Python  — données du problème (même convention que
                                 build_initial_tableau)
    max_iter  : int             — garde-fou contre les cycles

    Retourne
    --------
    SimplexResult  (même structure que solve())
    """
    result = SimplexResult()

    # Construction du tableau initial
    st = build_initial_tableau(c, A, b)
    n  = st.num_vars

    print("=" * 55)
    print("   RÉSOLUTION PAR LA MÉTHODE DU SIMPLEXE")
    print("=" * 55)

    for iteration in range(max_iter):
        print(f"\n{'━'*55}")
        print(f"  Itération {iteration}")
        print(f"{'━'*55}")

        # --- Choix de la variable entrante -------------------------------
        entering_col = choose_entering_variable(st, verbose=True)

        if entering_col == -1:
            # Tous les coefficients de la ligne z sont >= 0 → optimal
            print_tableau(st, title=f"Tableau — itération {iteration}")
            sol = extract_solution(st, verbose=True)

            result.status        = "optimal"
            result.optimal_value = sol["Z"]
            result.solution      = [sol["variables"][st.var_names[j]] for j in range(n)]
            result.message       = (
                f"Solution optimale atteinte en {iteration} itération(s)."
            )
            return result

        # --- Choix de la variable sortante (test du ratio) ---------------
        leaving_row = choose_leaving_variable(st, entering_col, verbose=True)

        if leaving_row == -1:
            result.status  = "non borné"
            result.message = "Problème non borné détecté."
            print("=" * 55)
            print("  PROBLÈME NON BORNÉ — arrêt.")
            print("=" * 55)
            return result

        # --- Tableau avec mise en évidence (colonne ↑, ligne →, pivot []) -
        print_tableau(st,
                      title=f"Tableau — itération {iteration}",
                      entering_col=entering_col,
                      leaving_row=leaving_row)

        # --- Pivot -------------------------------------------------------
        st = perform_pivot(st, leaving_row, entering_col, verbose=True)
        result.iterations.append([row[:] for row in st.tableau])

    # Dépassement du nombre maximal d'itérations
    result.status  = "infaisable"
    result.message = f"Nombre maximum d'itérations ({max_iter}) atteint."
    print(f"\n  ✗ {result.message}")
    return result


# ---------------------------------------------------------------------------
# Algorithme principal (numpy, sans affichage — utilisé par la GUI)
# ---------------------------------------------------------------------------

def solve(problem: ProblemData) -> SimplexResult:
    """
    Résout le problème de PL par la méthode du simplexe.

    Retourne un objet SimplexResult avec le statut, la valeur optimale
    et les valeurs des variables de décision.
    """
    result = SimplexResult()

    tableau = build_tableau(problem)
    result.iterations.append(tableau.copy())

    n, m = problem.num_vars, problem.num_constraints
    max_iter = 100

    for _ in range(max_iter):
        col = find_pivot_column(tableau)
        if col == -1:
            # Solution optimale atteinte
            result.status = "optimal"
            result.optimal_value = tableau[-1, -1]

            # Extraction de la solution
            solution = np.zeros(n)
            for j in range(n):
                col_vals = tableau[:-1, j]
                if np.sum(col_vals == 1) == 1 and np.sum(col_vals == 0) == m - 1:
                    row_idx = int(np.argmax(col_vals))
                    solution[j] = tableau[row_idx, -1]
            result.solution = solution
            result.message = "Solution optimale trouvée."
            return result

        row = find_pivot_row(tableau, col)
        if row == -1:
            result.status = "non borné"
            result.message = "Le problème est non borné."
            return result

        tableau = pivot(tableau, row, col)
        result.iterations.append(tableau.copy())

    result.status = "infaisable"
    result.message = "Nombre maximum d'itérations atteint."
    return result


# ---------------------------------------------------------------------------
# Démo rapide (python simplexe.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Problème exemple :
    #   Maximiser   5x1 + 4x2
    #   Sous         6x1 + 4x2 <= 24
    #                x1 + 2x2 <=  6
    #               -x1 +  x2 <=  1
    #                x1, x2   >=  0

    c_demo = [5, 4]
    A_demo = [
        [6,  4],
        [1,  2],
        [-1, 1],
    ]
    b_demo = [24, 6, 1]

    solve_steps(c_demo, A_demo, b_demo)

