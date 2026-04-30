"""
main.py
-------
Point d'entrée de l'application.
Lance l'interface graphique ou, en mode ligne de commande,
résout directement un exemple codé en dur.

Usage :
  python main.py          → ouvre la fenêtre graphique
  python main.py --cli    → résolution en mode console (exemple intégré)
"""

import sys

from simplexe import ProblemData, solve


def run_gui():
    """Lance l'interface graphique Tkinter."""
    from gui import SimplexeApp
    app = SimplexeApp()
    app.mainloop()


def run_cli_example():
    """
    Résout un exemple simple en ligne de commande et affiche le résultat.

    Problème exemple :
      Maximiser   5x1 + 4x2
      Sous         6x1 + 4x2 <= 24
                   x1 + 2x2 <=  6
                  -x1 +  x2 <=  1
                   x1, x2   >= 0
    """
    c = [5, 4]
    A = [
        [6, 4],
        [1, 2],
        [-1, 1],
    ]
    b = [24, 6, 1]

    problem = ProblemData(c, A, b)
    result = solve(problem)

    print("=" * 40)
    print("  Solveur Simplexe — mode console")
    print("=" * 40)
    print(f"Statut  : {result.status}")
    print(f"Message : {result.message}")

    if result.status == "optimal":
        print(f"Z opt   : {result.optimal_value:.4f}")
        for i, val in enumerate(result.solution):
            print(f"  x{i+1} = {val:.4f}")
        print(f"Itérations : {len(result.iterations) - 1}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli_example()
    else:
        run_gui()
