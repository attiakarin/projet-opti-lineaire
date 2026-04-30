"""
primal_dual.py
--------------
Methode primal-dual pour la programmation lineaire.

Probleme primal (P) :
    max  c^T x
    s.c. A x <= b,  x >= 0

Probleme dual (D) associe :
    min  b^T y
    s.c. A^T y >= c,  y >= 0

Theoreme de dualite forte (von Neumann) :
    Si (P) a une solution optimale x*, alors (D) a une solution optimale y*
    et  c^T x* = b^T y*.

Conditions de complementarite (KKT) :
    y_i * (b_i - (Ax*)_i) = 0   pour tout i  (ecarts primaux)
    x_j * ((A^T y*)_j - c_j) = 0  pour tout j  (ecarts duaux)

Notre algorithme :
-
1. Resoudre le primal par le simplexe (on obtient x* et le tableau final).
2. Lire les variables duales y* dans la ligne objectif du tableau final :
   y_i = coefficient de la variable d'ecart e_i dans la ligne f finale.
   (Par la theorie du simplexe : les coeff de la ligne f aux colonnes des
    variables d'ecart sont exactement les multiplicateurs duaux.)
3. Verifier les conditions de complementarite et afficher l'interpretation.
4. Verifier la dualite forte : c^T x* == b^T y*.
"""

import io
import contextlib

from simplexe import (
    SimplexTableau, SimplexResult,
    build_initial_tableau, solve_steps,
    print_tableau, extract_solution,
    choose_entering_variable, choose_leaving_variable, perform_pivot,
)


def solve_primal_dual(c, A, b, max_iter: int = 100, verbose: bool = True) -> SimplexResult:
    """
    Resout max c^T x  s.c.  Ax <= b,  x >= 0  et fournit l'analyse
    primal-dual complete (solution primale, solution duale, complementarite,
    dualite forte).
    """
    result = SimplexResult()
    n = len(c)
    m = len(b)

    if verbose:
        print("=" * 57)
        print("   METHODE PRIMAL-DUAL")
        print("=" * 57)
        print()
        print("  Probleme primal (P) :")
        print(f"    max  c^T x   avec c = {c}")
        print(f"    s.c. Ax <= b  avec b = {b}")
        print()
        print("  Probleme dual (D) :")
        print(f"    min  b^T y")
        print(f"    s.c. A^T y >= c,  y >= 0")
        print()

    # ETAPE 1 : Resoudre le primal par le simplexe
    if verbose:
        print("-" * 57)
        print("  ETAPE 1 : Resolution du probleme primal")
        print("-" * 57)
        print()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        primal_result = solve_steps(c, A, b, max_iter=max_iter)
    primal_output = buf.getvalue()

    if verbose:
        for line in primal_output.splitlines():
            print("    " + line)
        print()

    if primal_result.status != "optimal":
        result.status  = primal_result.status
        result.message = f"Primal : {primal_result.message}"
        if verbose:
            print(f"  Le probleme primal est {primal_result.status} -- arret.")
        return result

    x_star   = primal_result.solution
    Z_primal = primal_result.optimal_value

    # ETAPE 2 : Extraire les multiplicateurs duaux depuis le tableau final
    st_final = build_initial_tableau(c, A, b)
    for _ in range(max_iter):
        ec = choose_entering_variable(st_final, verbose=False)
        if ec == -1:
            break
        lr = choose_leaving_variable(st_final, ec, verbose=False)
        if lr == -1:
            break
        st_final = perform_pivot(st_final, lr, ec, verbose=False)

    obj_row = st_final.tableau[-1]
    y_star  = [obj_row[n + i] for i in range(m)]
    Z_dual  = sum(b[i] * y_star[i] for i in range(m))

    if verbose:
        print("-" * 57)
        print("  ETAPE 2 : Lecture des multiplicateurs duaux")
        print("-" * 57)
        print("  y_i = coeff de e_i dans la ligne objectif finale")
        print()
        for i in range(m):
            print(f"    y{i+1}  =  {y_star[i]:.4f}")
        print()
        print(f"  Valeur duale  b^T y*  =  {Z_dual:.4f}")
        print()

    # ETAPE 3 : Conditions de complementarite
    Ax_star = [sum(A[i][j] * x_star[j] for j in range(n)) for i in range(m)]
    AT_y    = [sum(A[i][j] * y_star[i] for i in range(m)) for j in range(n)]

    if verbose:
        print("-" * 57)
        print("  ETAPE 3 : Conditions de complementarite (KKT)")
        print("-" * 57)
        print()
        print("  (a) Complementarite primale : y_i * (b_i - (Ax*)_i) = 0")
        for i in range(m):
            slack_p = b[i] - Ax_star[i]
            product = y_star[i] * slack_p
            ok = abs(product) < 1e-8
            print(f"    i={i+1}: y{i+1}={y_star[i]:.4f}, slack={slack_p:.4f}, "
                  f"produit={product:.6f}  [{'OK' if ok else 'VIOLATION'}]")
        print()
        print("  (b) Complementarite duale : x_j * ((A^T y*)_j - c_j) = 0")
        for j in range(n):
            slack_d = AT_y[j] - c[j]
            product = x_star[j] * slack_d
            ok = abs(product) < 1e-8
            print(f"    j={j+1}: x{j+1}={x_star[j]:.4f}, slack={slack_d:.4f}, "
                  f"produit={product:.6f}  [{'OK' if ok else 'VIOLATION'}]")
        print()

    # ETAPE 4 : Dualite forte
    gap = abs(Z_primal - Z_dual)

    if verbose:
        print("-" * 57)
        print("  ETAPE 4 : Dualite forte (theoreme de von Neumann)")
        print("-" * 57)
        print()
        print(f"  Valeur primale  c^T x*  =  {Z_primal:.4f}")
        print(f"  Valeur duale    b^T y*  =  {Z_dual:.4f}")
        print(f"  Ecart de dualite        =  {gap:.8f}")
        if gap < 1e-6:
            print("  => Ecart nul : dualite forte verifiee !")
        else:
            print("  => ATTENTION : ecart non nul")
        print()

    result.status        = "optimal"
    result.optimal_value = Z_primal
    result.solution      = x_star
    result.message       = "Solution optimale trouvee par la methode primal-dual."

    if verbose:
        bar = "=" * 49
        print(bar)
        print("  Solution optimale (primal-dual)")
        print(bar)
        print("  Variables primales :")
        for j in range(n):
            print(f"    x{j+1}  =  {x_star[j]:.4f}")
        print("  Variables duales :")
        for i in range(m):
            print(f"    y{i+1}  =  {y_star[i]:.4f}")
        print(f"  Z  =  {Z_primal:.4f}")
        print(bar)
        print()

    return result


if __name__ == "__main__":
    c = [5, 4]
    A = [[6, 4], [1, 2], [-1, 1]]
    b = [24, 6, 1]
    result = solve_primal_dual(c, A, b, verbose=True)
    print(f"  Statut  : {result.status}")
    print(f"  Message : {result.message}")
