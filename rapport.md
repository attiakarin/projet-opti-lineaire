# Rapport de TP — Optimisation Linéaire
**ING1-GIA — Semestre 2 — 2025/2026**  
**TP Durée 4h30**

---

## 1. Introduction

Ce TP a pour objectif d'implémenter, en Python, plusieurs méthodes de résolution de problèmes de programmation linéaire (PL) sous la forme standard :

$$\max Z = c^T x \quad \text{sous} \quad Ax \leq b, \; x \geq 0$$

Nous avons développé trois méthodes :
- La **méthode du simplexe** (quand l'origine est admissible)
- La **méthode des deux phases** (quand l'origine n'est pas admissible)
- La **méthode primal-dual** (approche par le dual)

Le tout est accompagné d'une **interface graphique** (Tkinter) permettant la saisie des données, la résolution automatique, le mode pas à pas, et l'export des traces.

---

## 2. Méthode du simplexe

### 2.1 Principe général

La méthode du simplexe est un algorithme itératif qui parcourt les sommets du polyèdre des solutions admissibles jusqu'à trouver l'optimum. Elle repose sur le fait que si une solution optimale existe, elle se trouve nécessairement à un sommet (point extrême) du domaine réalisable.

On part d'un sommet initial (ici l'origine, donc x = 0 qui est admissible quand b ≥ 0), puis on se déplace vers un sommet adjacent qui améliore la fonction objectif, jusqu'à ce qu'aucune amélioration ne soit possible.

### 2.2 Construction du tableau initial

On part du problème sous forme standard. Pour chaque contrainte `a_i · x ≤ b_i`, on introduit une **variable d'écart** `e_i ≥ 0` telle que :

$$a_i \cdot x + e_i = b_i$$

On obtient un système avec `n` variables de décision et `m` variables d'écart. La base initiale est formée par les variables d'écart `{e_1, e_2, ..., e_m}`, ce qui correspond au sommet origine `x = 0`.

Le tableau a la forme `(m+1) × (n + m + 1)` :

```
      x1    x2   ...   e1    e2   ...  |   b
f  |  -c1   -c2  ...   0     0   ...  |   0
e1 |  a11   a12  ...   1     0   ...  |   b1
e2 |  a21   a22  ...   0     1   ...  |   b2
...
```

La ligne `f` contient les opposés des coefficients de l'objectif (−c), car on cherche à maximiser.

### 2.3 Critère d'entrée en base (variable entrante)

À chaque itération, on examine la ligne objectif `f`. Un coefficient négatif `c_j < 0` signifie que faire entrer la variable `x_j` en base **améliore** la valeur de Z.

On choisit la colonne dont le coefficient est **le plus négatif** (règle du gradient maximum) :

$$j^* = \arg\min_j \{ c_j \text{ dans la ligne } f \}$$

Si tous les coefficients sont ≥ 0, la solution courante est **optimale**.

### 2.4 Critère de sortie de base (test du ratio)

Une fois la variable entrante `x_{j*}` choisie, on détermine quelle variable sort de la base. On effectue le **test du ratio** : pour chaque ligne `i` avec `a_{i,j*} > 0`, on calcule :

$$\theta_i = \frac{b_i}{a_{i,j*}}$$

La variable sortante correspond à la ligne `i*` de ratio **minimum** :

$$i^* = \arg\min_i \left\{ \frac{b_i}{a_{i,j*}} \;|\; a_{i,j*} > 0 \right\}$$

Ce critère garantit que la nouvelle solution reste admissible (tous les b_i restent ≥ 0). Si aucun `a_{i,j*}` n'est strictement positif, le problème est **non borné**.

### 2.5 Opération de pivot

Le pivot transforme la base : `x_{j*}` entre, la variable de la ligne `i*` sort. On normalise la ligne pivot (diviser par l'élément pivot), puis on élimine la colonne `j*` dans toutes les autres lignes par combinaisons linéaires. Après le pivot, la colonne `j*` contient uniquement des 0 sauf un 1 sur la ligne `i*` (forme canonique).

### 2.6 Détection des cas particuliers

- **Solution optimale** : tous les coefficients de la ligne `f` sont ≥ 0.
- **Problème non borné** : la variable entrante choisie n'a aucun coefficient strictement positif dans les contraintes → le problème n'a pas de solution finie.

### 2.7 Exemple détaillé d'une itération

Considérons le problème : `max Z = 5x1 + 4x2` sous `6x1 + 4x2 ≤ 24`, `x1 + 2x2 ≤ 6`, `-x1 + x2 ≤ 1`.

**Tableau initial :**
```
          x1     x2     e1     e2     e3  |      b
f   |   -5.00  -4.00   0.00   0.00   0.00  |   0.00
─────────────────────────────────────────────────────
e1  |    6.00   4.00   1.00   0.00   0.00  |  24.00
e2  |    1.00   2.00   0.00   1.00   0.00  |   6.00
e3  |   -1.00   1.00   0.00   0.00   1.00  |   1.00
```

**Itération 1 :**
- Variable entrante : `x1` (coefficient le plus négatif : −5)
- Test du ratio : 24/6 = 4, 6/1 = 6, −1 ignoré → minimum = 4 → ligne `e1`
- Variable sortante : `e1`
- Pivot : élément `tableau[0][0] = 6.00`
- On divise la ligne 0 par 6, puis on élimine la colonne 0 dans toutes les autres lignes.

**Tableau après pivot :**
```
          x1     x2     e1     e2     e3  |      b
f   |    0.00  -0.67   0.83   0.00   0.00  |  20.00
─────────────────────────────────────────────────────
x1  |    1.00   0.67   0.17   0.00   0.00  |   4.00
e2  |    0.00   1.33  -0.17   1.00   0.00  |   2.00
e3  |    0.00   1.67   0.17   0.00   1.00  |   5.00
```

Z est passé de 0 à 20. On continue jusqu'à l'optimum Z* = 21 (x1 = 3, x2 = 1.5).

---

## 3. Méthode des deux phases

### 3.1 Pourquoi une deuxième méthode ?

Le simplexe classique suppose que l'origine est une solution de base admissible, c'est-à-dire que tous les b_i ≥ 0. Si ce n'est pas le cas (contraintes d'égalité ou de type ≥), on ne dispose pas d'une base initiale évidente. La méthode des deux phases contourne ce problème.

### 3.2 Phase 1 : trouver une base admissible

On ajoute des **variables artificielles** `a_1, a_2, ..., a_m` à chaque contrainte :

$$A x + e + a = b \quad (\text{après normalisation : } b_i \geq 0)$$

Les variables artificielles forment la base initiale. On construit un **problème auxiliaire** dont l'objectif est de **minimiser** la somme des variables artificielles :

$$\min w = a_1 + a_2 + \cdots + a_m$$

Si la valeur optimale de `w` est nulle (`w* = 0`), cela signifie qu'on a trouvé une base admissible pour le problème original (toutes les artificielles sont sorties de la base ou nulles). Si `w* > 0`, le **problème est infaisable** (le domaine réalisable est vide).

### 3.3 Phase 2 : résoudre le problème initial

On supprime les colonnes des variables artificielles du tableau obtenu à la fin de la Phase 1, on reconstruit la ligne objectif avec les vrais coefficients `−c`, et on repart avec la base admissible trouvée. On applique ensuite le simplexe standard jusqu'à l'optimum.

### 3.4 Exemple

Problème : `max Z = 2x1 + 3x2` sous `x1 + x2 ≥ 4`, `x1 + 3x2 ≤ 6`.

La contrainte `x1 + x2 ≥ 4` s'écrit `−x1 − x2 ≤ −4`, soit `b_1 = −4 < 0`. L'origine n'est pas admissible. On normalise en multipliant la ligne par −1 : `x1 + x2 ≥ 4 → x1 + x2 − e1 + a1 = 4`.

**Phase 1** résout `min w = a1 + a2`. À l'optimum `w* = 0`, on obtient la base `{x1, x2}` avec x1 = 3, x2 = 1.  
**Phase 2** optimise le vrai objectif : **Z* = 9** (x1 = 3, x2 = 1).

---

## 4. Méthode primal-dual

### 4.1 Dualité en programmation linéaire

À tout problème primal `max c^T x, Ax ≤ b, x ≥ 0` correspond un **problème dual** :

$$\min b^T y \quad \text{sous} \quad A^T y \geq c, \; y \geq 0$$

Les **conditions d'optimalité KKT** stipulent qu'aux points optimaux primal et dual, les conditions de complémentarité sont satisfaites :

$$y_i (b_i - a_i^T x) = 0 \quad \text{et} \quad x_j (a_j^T y - c_j) = 0$$

### 4.2 Principe de l'algorithme primal-dual

L'algorithme primal-dual maintient simultanément une solution duale admissible et cherche à construire une solution primale admissible. À chaque étape :

1. On part d'une solution duale admissible `y` (souvent `y = 0`).
2. On détermine les **variables primales libres** : `x_j` peut être > 0 seulement si `a_j^T y = c_j` (contrainte duale serrée).
3. On résout un **sous-problème primal restreint** (avec seulement les variables libres) pour tester si une solution primale admissible existe.
4. Si oui : on a trouvé l'optimum (conditions KKT satisfaites).
5. Sinon : on améliore la solution duale (on fait monter y) et on recommence.

### 4.3 Notre implémentation

Notre implémentation (`primal_dual.py`) utilise une approche simplifiée mais rigoureuse :

1. On construit le dual du problème et on démarre avec `y = 0`.
2. On identifie l'ensemble des indices "serrés" `J = { j : c^T_j - y^T A_j = 0 }` (les colonnes où la contrainte duale est active).
3. On résout le sous-problème primal restreint par le simplexe.
4. On extrait les variables duales mises à jour via les multiplicateurs du tableau final.
5. On répète jusqu'à admissibilité primale complète ou détection d'infaisabilité.

---

## 5. Interface graphique

### 5.1 Architecture

L'interface est développée avec **Tkinter** (bibliothèque standard Python). Elle est organisée en deux panneaux :
- **Panneau gauche** : saisie des données (coefficients de l'objectif `c`, matrice des contraintes `A`, second membre `b`), choix de la méthode, boutons de contrôle.
- **Panneau droit** : zone de texte scrollable avec coloration syntaxique pour l'affichage des tableaux et des résultats.

### 5.2 Fonctionnalités implémentées

| Fonctionnalité | Description |
|---|---|
| Saisie des données | Entrées texte pour c, A (une ligne par contrainte), b |
| Choix de méthode | Boutons radio : Simplexe / Deux phases / Primal-dual |
| Résolution complète | Bouton "Résoudre" — affiche toutes les itérations |
| Mode pas à pas | Bouton "Pas à pas" puis "Suivant" — une itération à la fois |
| Mise en évidence | Variable entrante `↑`, variable sortante `→`, pivot `[val]` |
| Export de traces | Sauvegarde tout l'historique dans un fichier `.txt` |
| Réinitialisation | Remet l'interface à zéro |

### 5.3 Mise en évidence du pivot

À chaque itération (en résolution complète ou pas à pas), le tableau affiché met visuellement en évidence :
- La **colonne entrante** : le nom de la variable est précédé de `↑`
- La **ligne sortante** : la ligne est précédée de `→`
- La **cellule pivot** : la valeur est entourée de crochets `[6.00]`

Exemple :
```
          ↑x1     x2     e1  |    b
f   |   -5.00  -4.00   0.00  |  0.00
─────────────────────────────────────
→ e1  | [6.00]   4.00   1.00  | 24.00
  e2  |   1.00   2.00   0.00  |  6.00
```

---

## 6. Structure du code

```
tp/
├── main.py          — Point d'entrée (GUI ou CLI avec --cli)
├── simplexe.py      — Méthode du simplexe (tableau, pivot, solve_steps)
├── two_phase.py     — Méthode des deux phases
├── primal_dual.py   — Méthode primal-dual
└── gui.py           — Interface graphique Tkinter
```

Chaque module est **indépendant** et peut être exécuté directement pour tester l'exemple intégré (ex : `python simplexe.py`, `python two_phase.py`, `python primal_dual.py`).

Les structures de données principales :

- **`SimplexTableau`** : liste de listes (tableau du simplexe), liste explicite de la base, noms des variables.
- **`SimplexResult`** : statut, valeur optimale, solution, nombre d'itérations, message.
- **`ProblemData`** : données brutes du problème (numpy arrays).

---

## 7. Analyse des résultats

### 7.1 Exemple 1 — Simplexe (origine admissible)

**Problème :** `max Z = 5x1 + 4x2` sous `6x1 + 4x2 ≤ 24`, `x1 + 2x2 ≤ 6`, `-x1 + x2 ≤ 1`

| Itération | Base | Z |
|---|---|---|
| 0 | {e1, e2, e3} | 0 |
| 1 | {x1, e2, e3} | 20 |
| 2 | {x1, x2, e3} | **21** |

**Solution optimale :** x1 = 3, x2 = 1.5, **Z* = 21**

La convergence est rapide (2 itérations). Chaque pivot améliore strictement Z.

### 7.2 Exemple 2 — Deux phases (origine non admissible)

**Problème :** `max Z = 2x1 + 3x2` sous `x1 + x2 ≥ 4`, `x1 + 3x2 ≤ 6`

- **Phase 1** (2 itérations) : w* = 0 → base admissible trouvée : {x1, x2}
- **Phase 2** (0 itérations) : déjà optimal après Phase 1

**Solution optimale :** x1 = 3, x2 = 1, **Z* = 9**

### 7.3 Exemple 3 — Primal-dual

**Problème :** `max Z = 5x1 + 4x2` (même exemple)

La méthode primal-dual converge vers le même optimum Z* = 21, en confirmant les conditions KKT à la solution finale.

---

## 8. Méthode de Séparation et Évaluation (Branch & Bound)

### 8.1 Principe

La méthode de **séparation et évaluation** (Branch & Bound, B&B) résout les problèmes de **programmation linéaire en nombres entiers** (PLNE). L'idée générale est :

1. **Relaxation linéaire** : résoudre le problème sans contrainte d'intégrité (PL standard). Si la solution est entière, c'est l'optimum global. Sinon, une variable fractionnaire `x_j = v` est choisie pour **brancher**.
2. **Séparation** : créer deux sous-problèmes fils en ajoutant l'une des contraintes :
   - Fils gauche : `x_j ≤ ⌊v⌋`
   - Fils droit  : `x_j ≥ ⌈v⌉` (revient à `-x_j ≤ -⌈v⌉`, nécessite la méthode des deux phases)
3. **Élagage** : un nœud est élagué si la relaxation est infaisable, non bornée, ou si sa borne supérieure est inférieure ou égale au meilleur entier connu.
4. **Mise à jour** : si la relaxation donne une solution entière meilleure, on l'enregistre comme nouvel optimum courant.

### 8.2 Règles implémentées

**Règle de choix de la variable à brancher :**
- `most_fractional` (par défaut) : choisir la variable dont la valeur est la plus proche de 0,5 (la plus « fractionnaire »). Cette règle tend à équilibrer l'arbre.
- `random` : choisir aléatoirement parmi les variables fractionnaires.

**Règle de sélection du nœud à explorer :**
- `best_bound` (par défaut) : explorer en priorité le nœud avec la borne supérieure la plus haute (stratégie optimiste). Converge souvent plus vite vers l'optimum.
- `deepest` : explorer en priorité le nœud le plus profond (stratégie DFS). Trouve des solutions entières plus rapidement mais peut explorer beaucoup de nœuds inutiles.

### 8.3 Exemple d'application

**Problème :** `max Z = 5x1 + 4x2` sous les contraintes  
`6x1 + 4x2 ≤ 24`, `x1 + 2x2 ≤ 6`, `-x1 + x2 ≤ 1`, `x1, x2 ∈ Z+`

| Nœud | Contrainte ajoutée | Z_relax | Solution relax. | Entier ? |
|------|--------------------|---------|-----------------|----------|
| 1    | (racine)           | 21,00   | (3; 1,5)        | Non      |
| 2    | x2 ≤ 1             | 20,67   | (3,33; 1)       | Non      |
| 3    | x2 ≥ 2             | 18,00   | (2; 2)          | **Oui** → Z*=18 |
| 4    | x2 ≤ 1, x1 ≤ 3    | 19,00   | (3; 1)          | **Oui** → Z*=19 |
| 5    | x2 ≤ 1, x1 ≥ 4    | 20,00   | (4; 0)          | **Oui** → Z*=20 |

**Solution optimale entière :** x1* = 4, x2* = 0, **Z* = 20** (vs. 21 pour la relaxation continue).

### 8.4 Intégration dans l'interface

Le module `branch_and_bound.py` est intégré dans `gui.py` via le bouton radio **"Branch & Bound (PLNE entier)"**. L'exploration complète de l'arbre est affichée dans la zone de sortie. Le mode pas à pas n'est pas disponible pour cette méthode (trop de sous-résolutions imbriquées).

---

## 9. Conclusion

Ce TP a permis d'implémenter et de comprendre en profondeur quatre méthodes de la programmation linéaire (et entière) :

1. **Le simplexe** est efficace lorsque l'origine est admissible. Il progresse de sommet en sommet en améliorant Z à chaque itération.
2. **La méthode des deux phases** généralise le simplexe à n'importe quel problème, en construisant d'abord une base de départ admissible.
3. **La méthode primal-dual** exploite la théorie de la dualité et les conditions KKT, offrant une perspective complémentaire sur l'optimalité.
4. **La méthode Branch & Bound** étend le simplexe aux problèmes entiers en explorant intelligemment un arbre de sous-problèmes, avec élagage des branches non prometteuses.

L'interface graphique développée rend l'utilisation accessible et pédagogique, avec la mise en évidence visuelle des pivots et le mode pas à pas pour comprendre chaque itération.

---

*Rapport rédigé dans le cadre du TP Optimisation Linéaire — ING1-GIA, Semestre 2, 2025/2026.*
