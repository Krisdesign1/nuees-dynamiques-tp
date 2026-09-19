# Références scientifiques et traçabilité

Cette résolution distingue explicitement les **éléments repris de la littérature** des **choix propres au projet**.

1. **Diday, E. (1971).** « Une nouvelle méthode en classification automatique et reconnaissance des formes : la méthode des nuées dynamiques ». *Revue de Statistique Appliquée*, 19(2), 19-33. Source principale pour le cadre des nuées, les étalons, l'alternance affectation/mise à jour et la discussion de convergence.
2. **MacQueen, J. (1967).** “Some Methods for Classification and Analysis of Multivariate Observations.” *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281-297. Référence historique pour k-means.
3. **Lloyd, S. P. (1982).** “Least Squares Quantization in PCM.” *IEEE Transactions on Information Theory*, 28(2), 129-137. Référence pour l'algorithme itératif centroïde-affectation.
4. **Gonzalez, T. F. (1985).** “Clustering to Minimize the Maximum Intercluster Distance.” *Theoretical Computer Science*, 38, 293-306. Inspiration de la stratégie farthest-first / maximin.
5. **Pearson, K. (1901).** “On Lines and Planes of Closest Fit to Systems of Points in Space.” *Philosophical Magazine*, 2(11), 559-572. Fondement historique des axes principaux.
6. **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. Référence pour les modèles gaussiens et la log-vraisemblance.
7. **Tukey, J. W. (1977).** *Exploratory Data Analysis*. Addison-Wesley. Référence pour les quartiles, l'IQR et les résumés robustes.
8. **Rousseeuw, P. J. (1987).** “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.” *Journal of Computational and Applied Mathematics*, 20, 53-65.
9. **Hubert, L. & Arabie, P. (1985).** “Comparing Partitions.” *Journal of Classification*, 2, 193-218. Référence pour l'Adjusted Rand Index.
10. **Harris, C. R. et al. (2020).** “Array Programming with NumPy.” *Nature*, 585, 357-362.

## Choix propres à cette résolution

Ne sont pas attribués à Diday : la standardisation automatique, la combinaison maximin + plusieurs redémarrages, la réparation des classes vides par scission du plus grand groupe, la moyenne des `q` prototypes les plus proches, la gaussienne **diagonale**, la structure par médiane/quartiles et le critère d'arrêt numérique. Ces éléments constituent la résolution logicielle développée pour ce TP.
