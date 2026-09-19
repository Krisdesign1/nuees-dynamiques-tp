# Références scientifiques et traçabilité

Ce fichier documente les sources utilisées par le projet. Les formulations du rapport sont paraphrasées ; lorsqu'un choix n'est pas directement tiré d'une publication, il est indiqué comme **choix propre au TP**.

1. **Diday, E. (1971).** *Une nouvelle méthode en classification automatique et reconnaissance des formes : la méthode des nuées dynamiques*. Revue de Statistique Appliquée, 19(2), 19-33. http://www.numdam.org/item?id=RSA_1971__19_2_19_0
   - Utilisé pour : cadre historique, étalons multiples, cycle itératif, fonctions D et R, convergence historique, discussion des distances.

2. **MacQueen, J. (1967).** *Some Methods for Classification and Analysis of Multivariate Observations*. Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Vol. 1, 281-297.
   - Utilisé pour : référence du k-means.

3. **Lloyd, S. P. (1982).** *Least Squares Quantization in PCM*. IEEE Transactions on Information Theory, 28(2), 129-137. https://doi.org/10.1109/TIT.1982.1056489
   - Utilisé pour : alternance affectation/mise à jour du cas ponctuel.

4. **Arthur, D., & Vassilvitskii, S. (2007).** *k-means++: The Advantages of Careful Seeding*. SODA 2007, 1027-1035. https://doi.org/10.1145/1283383.1283494
   - Utilisé pour : initialisation probabiliste inspirée de k-means++.

5. **Gonzalez, T. F. (1985).** *Clustering to Minimize the Maximum Intercluster Distance*. Theoretical Computer Science, 38, 293-306. https://doi.org/10.1016/0304-3975(85)90224-5
   - Utilisé pour : inspiration farthest-first de la sélection initiale de prototypes.

6. **Kaufman, L., & Rousseeuw, P. J. (1990).** *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. https://doi.org/10.1002/9780470316801
   - Utilisé pour : inspiration de type médoïde/PAM pour le raffinement local.

7. **Jolliffe, I. T., & Cadima, J. (2016).** *Principal Component Analysis: a Review and Recent Developments*. Philosophical Transactions of the Royal Society A, 374(2065), 20150202. https://doi.org/10.1098/rsta.2015.0202
   - Utilisé pour : axes principaux obtenus par valeurs propres de covariance.

8. **Mahalanobis, P. C. (1936).** *On the Generalised Distance in Statistics*. Proceedings of the National Institute of Sciences of India, 2(1), 49-55.
   - Utilisé pour : forme quadratique tenant compte de la covariance.

9. **Rousseeuw, P. J., & Croux, C. (1993).** *Alternatives to the Median Absolute Deviation*. Journal of the American Statistical Association, 88(424), 1273-1283. https://doi.org/10.1080/01621459.1993.10476408
   - Utilisé pour : MAD et facteur 1.4826.

10. **Rousseeuw, P. J. (1987).** *Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis*. Journal of Computational and Applied Mathematics, 20, 53-65. https://doi.org/10.1016/0377-0427(87)90125-7
    - Utilisé pour : coefficient de silhouette.

11. **Hubert, L., & Arabie, P. (1985).** *Comparing Partitions*. Journal of Classification, 2, 193-218. https://doi.org/10.1007/BF01908075
    - Utilisé pour : Adjusted Rand Index.

12. **Harris, C. R., et al. (2020).** *Array Programming with NumPy*. Nature, 585, 357-362. https://doi.org/10.1038/s41586-020-2649-2
    - Utilisé pour : environnement de calcul numérique NumPy.

## Choix propres au TP

Les éléments suivants ne sont pas attribués à une publication précise :

- interface unique `fit/cost` pour toutes les représentations ;
- combinaison exacte du coût des axes `distance orthogonale + alpha * distance le long des axes` ;
- règle de réparation des classes vides ;
- choix `min` ou `mean` pour la distance à une nuée de prototypes ;
- structure robuste exacte `médiane + MAD + somme des écarts normalisés` ;
- jeu de données synthétique, figures et tableaux expérimentaux.
