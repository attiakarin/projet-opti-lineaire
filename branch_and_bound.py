"""
branch_and_bound.py
-------------------
Methode de Separation et Evaluation (Branch & Bound) pour la
programmation lineaire en nombres entiers (PLNE).

Probleme :
    max  c^T x
    s.c. A x <= b,  x >= 0,  x entier

Algorithme general :
1. Partir avec le probleme initial (non traite).
2. Tant qu'il existe des problemes non traites :
   2.1 Choisir un probleme non traite selon la regle de noeud.
   2.2 Resoudre la relaxation LP associee.
   2.3 Si l'optimum relaxe est meilleur que le meilleur entier courant :
       - Si solution entiere : nouveau meilleur optimum entier.
       - Sinon : separer sur une variable fractionnaire.
   2.4 Marquer ce probleme comme traite.

Regles de selection du noeud (node_rule) :
  'best_bound' : noeud dont la relaxation LP a la plus grande valeur
  'deepest'    : noeud le plus profond dans l'arborescence (DFS)

Regles de choix de la variable de branchement (var_rule) :
  'most_fractional' : variable dont la partie fractionnaire est la plus
                      proche de 0.5 (plus grande distance a un entier)
  'random'          : variable non entiere choisie aleatoirement
"""

import math
import random
import io
import contextlib
from dataclasses import dataclass, field

from simplexe import SimplexResult, solve_steps
from two_phase import solve_two_phase


# ---------------------------------------------------------------------------
# Structures de donnees
# ---------------------------------------------------------------------------

@dataclass
class BBNode:
    """Noeud de l'arbre de Branch & Bound."""
    depth:        int
    c:            list
    A:            list
    b:            list
    parent_bound: float = float('inf')
    label:        str   = "Racine"


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def _solve_relaxation(c, A, b):
    """
    Resout la relaxation LP d'un noeud.
    Utilise le simplexe standard si b >= 0, sinon la methode des deux phases.
    Retourne (SimplexResult, texte_de_sortie).
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if any(bi < -1e-12 for bi in b):
            result = solve_two_phase(c, A, b, verbose=True)
        else:
            result = solve_steps(c, A, b)
    return result, buf.getvalue()


def _is_integer(val: float, tol: float = 1e-6) -> bool:
    """Retourne True si val est entiere a tol pres."""
    return abs(val - round(val)) < tol


def _choose_branch_var(solution, n: int, rule: str):
    """
    Retourne l'indice de la variable sur laquelle brancher.
    Retourne None si toutes les variables sont entieres.

    Regles :
      'most_fractional' : partie fractionnaire la plus proche de 0.5
      'random'          : choix aleatoire parmi les variables fractionnaires
    """
    fractional = [j for j in range(n) if not _is_integer(solution[j])]
    if not fractional:
        return None
    if rule == 'random':
        return random.choice(fractional)
    # Par defaut : most_fractional
    return min(fractional,
               key=lambda j: abs((solution[j] - math.floor(solution[j])) - 0.5))


# ---------------------------------------------------------------------------
# Algorithme principal
# ---------------------------------------------------------------------------

def solve_branch_and_bound(c, A, b,
                            var_rule:   str = 'most_fractional',
                            node_rule:  str = 'best_bound',
                            max_nodes:  int = 500,
                            verbose:   bool = True) -> SimplexResult:
    """
    Resout max c^T x  s.c. Ax <= b, x >= 0, x entier  par Branch & Bound.

    Parametres
    ----------
    c, A, b    : listes Python — donnees du probleme
    var_rule   : regle de branchement ('most_fractional' | 'random')
    node_rule  : regle de selection de noeud ('best_bound' | 'deepest')
    max_nodes  : nombre maximal de noeuds explores (garde-fou)
    verbose    : affichage detaille

    Retourne
    --------
    SimplexResult avec status, optimal_value, solution (entiere)
    """
    result = SimplexResult()
    n = len(c)

    best_int_val = -float('inf')
    best_int_sol = None
    nodes_explored = 0
    nodes_pruned   = 0

    # File des noeuds en attente
    pending = [BBNode(depth=0, c=list(c),
                      A=[row[:] for row in A], b=list(b),
                      parent_bound=float('inf'), label="Racine")]

    if verbose:
        print("=" * 60)
        print("   SEPARATION ET EVALUATION (Branch and Bound)")
        print("=" * 60)
        print(f"  Regle de noeud    : {node_rule}")
        print(f"  Regle de variable : {var_rule}")
        print(f"  Variables : {n},  Contraintes initiales : {len(b)}")
        print()

    node_counter = 0

    while pending and node_counter < max_nodes:
        node_counter += 1

        # ── Choix du prochain noeud ──────────────────────────────────────
        if node_rule == 'deepest':
            idx = max(range(len(pending)), key=lambda i: pending[i].depth)
        else:  # 'best_bound' (defaut)
            idx = max(range(len(pending)),
                      key=lambda i: pending[i].parent_bound)
        node = pending.pop(idx)
        nodes_explored += 1

        if verbose:
            print(f"{'─'*60}")
            print(f"  Noeud {node_counter}  |  profondeur={node.depth}"
                  f"  |  contrainte: {node.label}")
            print(f"{'─'*60}")

        # ── Resoudre la relaxation LP ────────────────────────────────────
        lp_result, lp_output = _solve_relaxation(node.c, node.A, node.b)

        if verbose:
            for line in lp_output.splitlines():
                print("    " + line)
            print()

        # ── Elagage : infaisable ou non borne ───────────────────────────
        if lp_result.status != "optimal":
            if verbose:
                print(f"  => Relaxation {lp_result.status} — noeud elague")
                print()
            nodes_pruned += 1
            continue

        z_relax = lp_result.optimal_value
        sol     = lp_result.solution

        if verbose:
            print(f"  => Z_relax = {z_relax:.4f}   "
                  f"(meilleur entier courant : "
                  f"{'aucun' if best_int_sol is None else f'{best_int_val:.4f}'})")

        # ── Elagage par borne ────────────────────────────────────────────
        if z_relax <= best_int_val + 1e-8:
            if verbose:
                print(f"  => Z_relax <= meilleur entier ({best_int_val:.4f})"
                      f" — noeud elague par borne")
                print()
            nodes_pruned += 1
            continue

        # ── Choisir la variable de branchement ───────────────────────────
        branch_var = _choose_branch_var(sol, n, var_rule)

        if branch_var is None:
            # ── Solution entiere ─────────────────────────────────────────
            if verbose:
                print(f"  => Solution entiere ! Z = {z_relax:.4f}")
                for j in range(n):
                    print(f"     x{j+1} = {sol[j]:.0f}")
                print()
            if z_relax > best_int_val + 1e-8:
                best_int_val = z_relax
                best_int_sol = sol[:]
                if verbose:
                    print(f"  *** Nouveau meilleur optimum entier : Z* = {best_int_val:.4f} ***")
                    print()
        else:
            # ── Separation ───────────────────────────────────────────────
            v       = sol[branch_var]
            floor_v = math.floor(v)
            ceil_v  = math.ceil(v)
            frac    = v - floor_v

            if verbose:
                print(f"  => x{branch_var+1} = {v:.6f}  "
                      f"(frac = {frac:.4f}, dist 0.5 = {abs(frac-0.5):.4f})")
                print(f"  => Separation : "
                      f"x{branch_var+1} <= {floor_v}   |   "
                      f"x{branch_var+1} >= {ceil_v}")
                print()

            # Fils gauche : x_j <= floor(v)
            row_left = [0.0] * n
            row_left[branch_var] = 1.0
            A_left = [row[:] for row in node.A] + [row_left]
            b_left = list(node.b) + [float(floor_v)]

            pending.append(BBNode(
                depth        = node.depth + 1,
                c            = node.c,
                A            = A_left,
                b            = b_left,
                parent_bound = z_relax,
                label        = f"x{branch_var+1} <= {floor_v}",
            ))

            # Fils droit : x_j >= ceil(v)  <=>  -x_j <= -ceil(v)
            row_right = [0.0] * n
            row_right[branch_var] = -1.0
            A_right = [row[:] for row in node.A] + [row_right]
            b_right = list(node.b) + [float(-ceil_v)]

            pending.append(BBNode(
                depth        = node.depth + 1,
                c            = node.c,
                A            = A_right,
                b            = b_right,
                parent_bound = z_relax,
                label        = f"x{branch_var+1} >= {ceil_v}",
            ))

    # ── Resume final ─────────────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  RESUME FINAL")
        print("=" * 60)
        print(f"  Noeuds explores : {nodes_explored}")
        print(f"  Noeuds elagués  : {nodes_pruned}")
        print()

    if best_int_sol is not None:
        result.status        = "optimal"
        result.optimal_value = best_int_val
        result.solution      = best_int_sol
        result.message       = f"Optimum entier trouve : Z* = {best_int_val:.4f}"

        if verbose:
            bar = "=" * 50
            print(bar)
            print("  Solution optimale entiere")
            print(bar)
            for j in range(n):
                print(f"  x{j+1}  =  {best_int_sol[j]:.0f}")
            print(f"  {'─'*20}")
            print(f"  Z*  =  {best_int_val:.4f}")
            print(bar)
            print()
    else:
        result.status  = "infaisable"
        result.message = "Aucune solution entiere admissible trouvee."
        if verbose:
            print("  Aucune solution entiere admissible trouvee.")
            print()

    return result


# ---------------------------------------------------------------------------
# Test rapide
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Exemple classique PLNE :
    # max  Z = 5x1 + 4x2
    # s.c. 6x1 + 4x2 <= 24
    #      x1  + 2x2 <=  6
    #     -x1  +  x2 <=  1
    #      x1, x2 entiers >= 0
    # Solution attendue : x1=3, x2=1, Z=19  (vs Z=21 en relaxation)

    c = [5, 4]
    A = [
        [ 6,  4],
        [ 1,  2],
        [-1,  1],
    ]
    b = [24, 6, 1]

    print("=== Test Branch & Bound (most_fractional, best_bound) ===")
    res = solve_branch_and_bound(c, A, b,
                                  var_rule='most_fractional',
                                  node_rule='best_bound',
                                  verbose=True)
    print(f"\nStatut  : {res.status}")
    print(f"Message : {res.message}")
