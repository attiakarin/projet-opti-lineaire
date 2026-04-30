"""
two_phase.py
------------
Méthode des deux phases pour les problèmes de PL où l'origine
n'est PAS une solution admissible (cas b_i < 0 après mise en forme,
ou contraintes avec >= / =).

Forme traitée (après conversion) :
  max Z = c^T x
  sous  A_eq x = b,  b >= 0,  x >= 0

Usage :
  from two_phase import solve_two_phase
  result = solve_two_phase(c, A, b, verbose=True)
"""

import io
import contextlib

from simplexe import (
    SimplexTableau,
    SimplexResult,
    print_tableau,
    choose_entering_variable,
    choose_leaving_variable,
    perform_pivot,
    extract_solution,
)


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _pivot_loop(st: SimplexTableau, label: str,
                forbidden_cols: list[int] | None = None,
                verbose: bool = True) -> tuple[SimplexTableau, str]:
    """
    Boucle simplexe générique utilisée par les deux phases.

    Paramètres
    ----------
    st            : tableau courant
    label         : préfixe affiché dans les titres ("Phase 1" / "Phase 2")
    forbidden_cols: colonnes à exclure du choix de la variable entrante
                    (variables artificielles en phase 2)
    verbose       : affichage itération par itération

    Retourne
    --------
    (SimplexTableau final, statut : "optimal" | "non borné" | "max_iter")
    """
    max_iter = 200
    for it in range(max_iter):
        if verbose:
            print(f"\n{'━'*55}")
            print(f"  {label} — itération {it}")
            print(f"{'━'*55}")
            print_tableau(st, title=f"Tableau {label} — itération {it}")

        entering_col = choose_entering_variable(
            st, verbose=verbose, forbidden_cols=forbidden_cols)
        if entering_col == -1:
            return st, "optimal"

        leaving_row = choose_leaving_variable(st, entering_col, verbose=verbose)
        if leaving_row == -1:
            return st, "non borné"

        if verbose:
            leaving_name = st.var_names[st.basis[leaving_row]]
            pivot_val    = st.tableau[leaving_row][entering_col]
            bar = "┄" * 45
            print(bar)
            print(f"  RÉSUMÉ DU PIVOT — {label} it. {it}")
            print(f"  Entrante : {st.var_names[entering_col]}")
            print(f"  Sortante : {leaving_name}")
            print(f"  Pivot    : {pivot_val:.4f}  "
                  f"(ligne {leaving_row}, col {entering_col})")
            print(bar)

        st = perform_pivot(st, leaving_row, entering_col, verbose=verbose)

    return st, "max_iter"


# ---------------------------------------------------------------------------
# Construction du tableau de Phase 1
# ---------------------------------------------------------------------------

def _build_phase1_tableau(A, b) -> SimplexTableau:
    """
    Construit le tableau augmenté de la Phase 1.

    On ajoute une variable artificielle a_i pour chaque contrainte.
    La fonction objectif de la Phase 1 est :

        min w = a_1 + a_2 + ... + a_m   ⟺   max -w = -(a_1+…+a_m)

    Structure des colonnes :
        x1…xn | e1…em | a1…am | b
    (les variables d'écart ei sont déjà présentes si b >= 0 ;
     ici on accepte des b quelconques en ajoutant systématiquement
     les artificelles comme variables de base initiale.)

    Gestion de b_i < 0 : on multiplie la ligne par -1 pour rendre b_i >= 0,
    ce qui change le sens du signe des coefficients de A sur cette ligne.
    """
    m = len(b)
    n = len(A[0])

    # Normalisation : b_i doit être >= 0
    A_norm = [row[:] for row in A]
    b_norm = b[:]
    for i in range(m):
        if b_norm[i] < 0:
            b_norm[i] = -b_norm[i]
            A_norm[i] = [-v for v in A_norm[i]]

    # Colonnes : x1…xn | e1…em | a1…am | b
    total_cols = n + m + m + 1
    var_names  = ([f"x{j+1}" for j in range(n)] +
                  [f"e{i+1}" for i in range(m)] +
                  [f"a{i+1}" for i in range(m)] +
                  ["b"])

    tableau = []
    for i in range(m):
        row = [0.0] * total_cols
        for j in range(n):
            row[j] = float(A_norm[i][j])
        # Contrainte originale b_i < 0 → ligne normalisée par -1 → contrainte >=
        # Variable d'écart : surplus (coeff -1) au lieu de slack (coeff +1)
        row[n + i]     = -1.0 if b[i] < 0 else 1.0
        row[n + m + i] = 1.0   # variable artificielle (toujours +1)
        row[-1]        = float(b_norm[i])
        tableau.append(row)

    # Ligne objectif Phase 1 : max -w = -(a_1+…+a_m)
    # Convention MAX : obj_row[j] = -c_j  →  c_{a_i} = -1  →  obj_row[a_i] = +1
    # Avant d'éliminer, la ligne est [0…0 | 0…0 | +1…+1 | 0]
    obj_row = [0.0] * total_cols
    for i in range(m):
        obj_row[n + m + i] = 1.0   # -c_{a_i} = -(-1) = +1
    tableau.append(obj_row)

    # Indices des variables en base initiale : les artificelles a1…am
    basis = [n + m + i for i in range(m)]

    st = SimplexTableau(tableau, basis, var_names, n, m)

    # Élimination initiale : rendre les colonnes de base canoniques dans la ligne z
    for i in range(m):
        col = n + m + i   # colonne de a_i
        factor = st.tableau[-1][col]  # = +1
        if abs(factor) > 1e-12:
            st.tableau[-1] = [
                st.tableau[-1][k] - factor * st.tableau[i][k]
                for k in range(total_cols)
            ]

    return st


# ---------------------------------------------------------------------------
# Méthode des deux phases
# ---------------------------------------------------------------------------

def solve_two_phase(c, A, b, verbose: bool = True) -> SimplexResult:
    """
    Résout un problème de PL par la méthode des deux phases.

    Paramètres
    ----------
    c : list[float]        — coefficients objectif (maximisation)
    A : list[list[float]]  — matrice des contraintes (m × n)
    b : list[float]        — second membre
    verbose : bool         — affichage détaillé

    Retourne
    --------
    SimplexResult

    Algorithme
    ----------
    Phase 1 :
      - Introduire des variables artificielles a_i pour chaque contrainte.
      - Résoudre  max -w = -(a_1+…+a_m).
      - Si  w* > ε  →  problème infaisable (les artificielles ne peuvent
        pas toutes être nulles).

    Phase 2 :
      - Supprimer les colonnes des variables artificielles du tableau.
      - Remplacer la ligne objectif par les vrais coefficients de c,
        puis éliminer les variables basiques pour rétablir la forme canonique.
      - Résoudre le problème initial par le simplexe standard.
    """
    result = SimplexResult()
    n = len(c)
    m = len(b)

    if verbose:
        print("=" * 55)
        print("   MÉTHODE DES DEUX PHASES")
        print("=" * 55)

    # ══════════════════════════════════════════════════════
    # PHASE 1
    # ══════════════════════════════════════════════════════
    if verbose:
        print("\n" + "─" * 55)
        print("  PHASE 1 : recherche d'une base admissible")
        print("─" * 55)

    st1 = _build_phase1_tableau(A, b)

    if verbose:
        print_tableau(st1, title="Tableau initial Phase 1")

    st1, status1 = _pivot_loop(st1, "Phase 1", verbose=verbose)

    # Valeur de w* = -( ligne objectif colonne b )
    w_star = -st1.tableau[-1][-1]

    if verbose:
        print(f"\n  → Phase 1 terminée  (statut = {status1})")
        print(f"  → Valeur de w* = {w_star:.6f}")

    if w_star > 1e-8:
        result.status  = "infaisable"
        result.message = f"Phase 1 : w* = {w_star:.6f} > 0 → problème infaisable."
        if verbose:
            print(f"\n  ✗ {result.message}")
        return result

    # ══════════════════════════════════════════════════════
    # PHASE 2 — construction du tableau
    # ══════════════════════════════════════════════════════
    if verbose:
        print("\n" + "─" * 55)
        print("  PHASE 2 : optimisation du problème initial")
        print("─" * 55)

    # Indices des colonnes artificielles dans st1
    art_cols = list(range(n + m, n + m + m))

    # Vérifier qu'aucune artificielle n'est en base
    # (si c'est le cas on tente de la faire sortir)
    basis2 = st1.basis[:]
    tab2   = [row[:] for row in st1.tableau]

    for row_idx, col_idx in enumerate(basis2):
        if col_idx in art_cols:
            # Chercher une colonne non-artificielle avec coeff non nul
            replaced = False
            for j in range(n + m):
                if abs(tab2[row_idx][j]) > 1e-9:
                    # Pivot pour faire sortir l'artificielle
                    pivot_val = tab2[row_idx][j]
                    tab2[row_idx] = [v / pivot_val for v in tab2[row_idx]]
                    for i in range(len(tab2)):
                        if i != row_idx:
                            f = tab2[i][j]
                            tab2[i] = [tab2[i][k] - f * tab2[row_idx][k]
                                       for k in range(len(tab2[0]))]
                    basis2[row_idx] = j
                    replaced = True
                    break
            if not replaced:
                # Ligne redondante — on la gardera (ne nuit pas)
                pass

    # Supprimer les colonnes artificielles et reconstruire les noms
    keep_cols = [j for j in range(len(st1.var_names) - 1) if j not in art_cols]
    keep_cols.append(len(st1.var_names) - 1)   # garder la colonne b

    new_tab = []
    for row in tab2:
        new_tab.append([row[k] for k in keep_cols])

    new_var_names = [st1.var_names[k] for k in keep_cols]

    # Remapper les indices de la base (les artificelles disparaissent)
    col_map = {old: new for new, old in enumerate(keep_cols)}
    new_basis = [col_map[b_idx] for b_idx in basis2]

    # Remplacer la ligne objectif par les vrais coefficients de c
    obj_row = [0.0] * len(new_var_names)
    for j in range(n):
        obj_row[j] = -float(c[j])   # max → on stocke -c
    new_tab[-1] = obj_row

    st2 = SimplexTableau(new_tab, new_basis, new_var_names, n, m)

    # Élimination initiale : rétablir la forme canonique sur la ligne objectif
    for row_idx, col_idx in enumerate(st2.basis):
        factor = st2.tableau[-1][col_idx]
        if abs(factor) > 1e-12:
            num_c = len(st2.tableau[0])
            st2.tableau[-1] = [
                st2.tableau[-1][k] - factor * st2.tableau[row_idx][k]
                for k in range(num_c)
            ]

    if verbose:
        print_tableau(st2, title="Tableau initial Phase 2")

    # ══════════════════════════════════════════════════════
    # PHASE 2 — résolution
    # ══════════════════════════════════════════════════════
    # Colonnes des artificielles (maintenant absentes → forbidden vide)
    st2, status2 = _pivot_loop(st2, "Phase 2", forbidden_cols=None,
                               verbose=verbose)

    if status2 == "non borné":
        result.status  = "non borné"
        result.message = "Phase 2 : problème non borné."
        if verbose:
            print(f"\n  ✗ {result.message}")
        return result

    # Extraction de la solution
    sol = extract_solution(st2, verbose=verbose)
    result.status        = "optimal"
    result.optimal_value = sol["Z"]
    result.solution      = [sol["variables"].get(f"x{j+1}", 0.0) for j in range(n)]
    result.message       = "Solution optimale trouvée par la méthode des deux phases."

    if verbose:
        print(f"\n  ✓ {result.message}")

    return result


# ---------------------------------------------------------------------------
# Démo rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Exemple classique où l'origine N'EST PAS admissible :
    #   max  Z = 2x1 + 3x2
    #   sous  x1 +  x2 >= 4   (contrainte >=  → -x1 - x2 <= -4)
    #          x1 + 3x2 <= 6
    #          x1, x2  >= 0
    #
    # Après conversion en <= :
    #   -x1 - x2 <= -4   (b_1 = -4 < 0 → nécessite variable artificielle)
    #    x1 + 3x2 <=  6

    c = [2, 3]
    A = [[-1, -1],
         [ 1,  3]]
    b = [-4, 6]

    solve_two_phase(c, A, b, verbose=True)
