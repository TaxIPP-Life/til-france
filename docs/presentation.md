# Introduction

![alt text][logo]

[logo]: https://avatars1.githubusercontent.com/u/4996093?v=3&s=200 "TaxIPP-Life logo"


## L'ambition de \til

### Un modèle de microsimulation généraliste

L'objectif du modèle \txl \ est l'étude du cycle de vie dans toutes ses dimensions.
Il se distingue en cela de la plupart des autres modèles de microsimulation dynamique qui se concentre sur un domaine particulier : l'éducation, les retraites, la dépendance ou l'activité féminine par exemple.

La généralité désirée du modèle pousse à ne faire aucune impasse sur les différentes étapes marquantes de la vie économique de la population.
\txl \ est ainsi par exemple, à ma connaissance, le seul modèle de microsimulation intégrant les héritages et droits de succession.

Prendre en compte le plus possible de dimensions permet aussi en théorie d'étudier des phénomènes plus complexe, d'un second ordre.
On peut penser à l'influence de l'assurance chômage sur les choix d'éducation, ou celle de l'allongement de la vie sur l'immobilier, les loyers et les allocations logements mais aussi à bien d'autres problématiques.

Cet objectif généraliste permet à chaque partie de profiter des avancées sur le reste du modèle.
La structure du modèle doit par exemple permettre qu'un travail sur les choix de départ en retraite profite aux études sur la consommation des personnes âgées.
L'investissement qui peut être fait sur les parties sensibles et clés du modèle comme le marché du travail ou la démographie est utile pour toutes les études et le spécialiste de la santé n'aura pas besoin de se spécialiser dans le marché du travail plus qu'il ne le souhaite.

### Mesurer les niveaux de vie
\txl \ part du principe que les études en coupe sont par définition limitée.
Tout le monde est conscient que le revenu d'une année ne peut-être qu'un proxy du niveau de vie.
Un revenu sur cycle de vie ne sera aussi qu'un proxy mais forcément de bien meilleure qualité.
On peut en effet faire trois principaux reproches au revenu en coupe.
Il fluctue d'une année sur l'autre, il n'a presque pas de sens dans les années d'inactivité, en particulier pendant les études et un revenu en coupe est toujours exposé au fait qu'il ne tient pas compte du patrimoine.

#### Les revenus peuvent varier d'une année à l'autre. Plusieurs éléments peuvent expliquer des variations de revenus d'une année à l'autre.
Des aléas professionnel (prime, bonne année, mauvaise récolte,?) avec éventuellement de changement de statut (période de chômage, réduction de l'activité suite à la naissance d'un enfant, congés sabbatiques, etc).
Si on croit que ces « chocs » sont lissés par les ménages alors la consommation d'une année n'est pas représentative des revenus au-delà de cette année.

#### Les revenus varient en fonction de la période de la vie.
Le point précédent commence à sous-entendre que l'année est une période trop courte pour prendre en compte les fluctuations de revenus, si on cherche quelle est la plus petite période, on se trouve obligé de prendre en compte les études et la retraite, on arrive rapidement à la conclusion que seul l'ensemble du cycle de vie permet d'approcher le niveau de vie (on se rapproche du revenu permanent de Lollivier- Verger).
En effet, les revenus d'un étudiant sont en général très faible alors que pendant ses études, l'étudiant est censé augmenté son revenu actualisé.
Quoi qu'il en soit, son revenu n'est pas du tout indicateur de son niveau de vie même si on sait que le lissage des revenus n'est pas parfait.
En général les étudiants sont exclus des enquêtes de l'Insee  sur les revenus, un aveu d'impuissance.
Pour les retraités, certaines théories de la justice affirment, qu'il n'y a pas de raison de comparer les revenus des retraités avec ceux des actifs. Le régime général français est trompeur.
Il y a de l'épargne forcée et un système par répartition, mais la vision classique de la retraite pendant laquelle on dépense le capital épargné pendant la période d'activité existe bel et bien.
Le niveau de vie des retraités doit prendre en compte les revenus de leur période d'activité et pas seulement le niveau des revenus pendant la retraite.
Les entrepreneurs représentent une autre catégorie au revenu annuel non significatif.
Si on considère que leur travail n'a pas vocation à leur fournir un revenu au quotidien mais à augmenter le patrimoine, la valeur, de leur entreprise pour la revendre alors, très schématiquement, ils reçoivent en une fois le revenu de toute leur vie.
On ne peut pas considérer qu'ils sont pauvres toutes les années sauf une où ils sont tout en haut de la distribution des revenus.

#### L'ambigüité patrimoine vs revenus
Le cas des retraités est révélateur d'un conflit face au quel se retrouve chaque économiste se penchant sur les inégalités c'est l'opposition entre le patrimoine et le revenu.
On peut décrire le phénomène des pauvres riches, incarnés magistralement par le fameux pêcheur de l'île de Ré, qui a un revenu faible, mais un patrimoine pouvant représenter plus que son revenu annuel sur cent ans.
Il n'est pas évident de classer cet individu dans la hiérarchie des niveaux de vie.
On peut inclure les plus-value-latente dans le revenu, y compris annuel.
Mais le cycle de vie permet aussi de tenir compte des transferts de patrimoine (en particulier les héritages) en les considérant comme des revenus.
On mesure ainsi le niveau de vie de quelqu'un en incluant le revenu qu'il peut tirer, par exemple, la revente de son patrimoine.

### Mesurer la redistribution

Puisqu'on ne mesure pas bien les niveaux de vie, il est évident qu'on ne mesure pas bien la redistribution qui modifie ces niveaux de vie
 Au-delà des points qui découlent directement des assertions précédentes (fixation sur l'état à un instant donné d'un individu, non prise en compte du patrimoine et donc des taxes qui lui sont associées, etc.), d'autres éléments limitent aussi le bien fondé des études en coupe.

### La redistribution within (intrapersonnelle)
Comme on l'a vu, les revenus fluctuent au cours de la vie, en conséquence, la participation au budget de l'état aussi.
On peut par exemple percevoir la prime pour l'emploi en début de carrière puis être contributeur net à l'impôt sur le revenu.
On peut aussi percevoir des allocations familiales et y cotiser avant l'arrivée d'un enfant et après les 21 ans du cadet.
Tant et si bien qu'une partie des prestations reçues par un individu sont en fait financées par lui-même à d'autres moments de sa vie. En voyant ces transferts comme allant d'un individu à l'autre on surestime la redistribution.

### Les transferts assurantiels
Un certain nombre de cotisation (maladie, vieillesse, chômage) correspondent à une assurance.
En coupe, on considère, et cela est bien pratique, qu'il s'agit d'une consommation, forcée certes mais d'une consommation qu'on ne peut donc pas envisagée comme de la redistribution.
Inversement, les revenus de ces assurances, sont eux aussi exclu de la redistribution car ils correspondent à une prime d'assurance et donc à cette consommation qui n'est pas de la redistribution.
Cependant, rien ne dit que les cotisations assurantielles financent bien le risque qu'elles couvrent.
On passe possiblement à côté d'élément redistributif.
Par exemple, rien ne dit, que certains ne financent pas plus la retraite ou les dépenses de maladie des autres.
On sous-estime peut-être la redistribution.

### Évaluation \textit{ex ante} de réformes

Un modèle de microsimulation est aussi un élément d'évaluation ex-ante des politiques publiques.
Les simulations en coupe sont en générale assez limitée.

D'abord elles sont le plus souvent statiques alors que l'on peut toujours imaginer des réactions comportementales aux réformes.
Ensuite, l'évaluation n'est pas obligée d'être cantonnée à l'année en cours ou à l'année suivante.
Dire qu'une réforme de l'IR ne touchera par exemple que les ménages imposables, c'est oublier que certains de ces ménages imposables, ne le seront plus dans les années à venir (enfants, perte de revenus) et certains le seront alors qu'ils ne le sont pas actuellement.
Avec une approche sur cycle de vie, on peut estimer l'impact de trois façon, ceux concernés à l'instant $t$, ceux qui seront concernés à $t$ après $t$, et ceux qui aurait été concernés si la réforme avait été en vigueur depuis leur naissance.
Il s'agit d'avoir une autre vision des statistiques, cohérente avec la vie des gens et un peu distante de leur présent.
Enfin, les simulations d'un modèle dynamique sur le cycle de vie, ont un champ des possibles très intéressant et on peut simuler un monde avec moins de séparation, une autre immigration, des modifications dans le prix des logements, moins de croissance, etc. et quantifier l'impact de ces réformes de sociétés.


## Historique du modèle

### Contexte lors de la création

On ne peut pas parler de l'historique de \txl \ sans parler des deux piliers sur lesquels s'est appuyé le modèle et qui étaient là avant lui : Liam2 et OpenFisca.
Le premier est développé pour la dynamique de la simulation, le second est développé pour le calcul de la législation, en particulier la législation française.
Il faut remercier ces deux projets, et le bon goût qu'ils ont eu d'être OpenSource.
Il faut aussi remercier les personnes qui étaient responsable de ces modèles, en 2012-13, à la naissance de \txl , respectivement Gaëtan de Menten et Mahdi Ben Jelloul.
De nombreuses interactions avec eux ont permis de profiter de leur expérience et de reprendre ces travaux pour la construction de \txl.
Ces échanges ont été très important pour \txl \ et on espère que les gains à l'échange n'ont pas été unilatéraux.
Liam2 et OpenFisca ont possiblement profité du regard extérieur sur leur travaux.
Il serait évidemment intéressant que les trois modèles continuent de se développer, et le plus collectivement possible.

Le modèle \txl \ a aussi eu la chance d'avoir comme prédécesseur le modèle Destinie de l'Insee.
Les documents publiés sur ce modèle ainsi que les échanges avec son équipe de développement ont permis d'avoir rapidement de bonnes équations de simulation.
En particulier, Didier Blanchet, parce qu'il avait commencé une version en R de Destinie, a permis à \txl \ de s'appuyer dès le début sur un code lisible et simple avec les grandes lignes d'un modèle de microsimulation dynamique.

### Frise chronologique
\label{historique
Le projet \txl \ a débuté en septembre 2012 à l'initiative d'Alexis Eidelman qui a passé un an à asseoir les bases du projet.
Avec Béatrice Boutchenik, une première étape a été l'appariement de l'enquête patrimoine avec les déclarations annuelles de données sociales (DADS)\footnote{On pourra lire son mémoire de master pour plus d'informations}.\\

Ensuite, l'intérêt s'est d'abord porté sur le moteur du modèle (voir partie \ref{technique}) en recherchant si possible un temps de calcul réduit.
L'expérience montre en effet que le temps de calcul d'une simulation a une influence sur les améliorations du modèle et sur la qualité des statistiques produites.
L'essentiel de la structure du modèle dont l'interface entre Liam2 et \of , ainsi que bon nombre d'amélioration dans ces deux modèle ont permis de donner une première version de \txl.
Cette version, avancée techniquement, n'en reste pas moins trop simple au niveau des équations de simulation et des résultats finaux.

Le travail sur les données d'abord réalisé en R est traduit en Python à l'été 2013.
A cette occasion, une optimisation de l'algorithme de matching permet de faire ce calcul très rapidement.
On peut désormais à cette étape ne plus se limiter et travailler sur la base étendue a près de 10 millions d'individus.
%
%Antoine Boiron effectue un stage à l'IPP entre aoûtt et octobre 2013.
%Il se penche plus profondément sur le matching entre les bases de l'enquête patrimoine et l'EIC.
%En dehors d'effectuer cette opération en Python, on peut désormais effectuer une augmentation de l'enquête patrimoine avant ce matching.

%\begin{tikzpicture}[snake=zigzag, line before snake = 5mm, line after snake = 5mm]
%%draw horizontal line   
%\draw (0,0) -- (12,0);
%%draw vertical lines
%\foreach \x in {0,1,12}
%   \draw (\x cm,3pt) -- (\x cm,-3pt);
%
%%draw nodes
%\draw (0,0) node[below=3pt] {$ Sept-12 $} node[above=3pt] {$ Début du projet  $};
%\draw (1,0) node[below=3pt] {$ Sept-12 $} node[above=3pt] {$ Début du projet  $};
%\draw (12,0) node[below=3pt] {$ Sept-13 $} node[above=3pt] {$ ??? $};
%
%\end{tikzpicture}

\begin{table}
\centering
\begin{tabular}{|c|c|c|c|c|c|}
\hline
? & \multicolumn{3}{c|}{Personne sur le projet} & Réalisation & Production \\
\hline
? & Alexis & Béatrice & ? & ? & ? \\
\hline
Sept 2012 & X & X & ? &  & ? \\
\hline
Dec 2012 & X & ? & ? & Matching Patrimoine EIR-EIC & ? \\
\hline
Mai 2013 & X & ? & ? & Connection OpenFisca - Liam & Présentation JEM \\
\hline
Juillet 2013 & X & ? & ? & ? & Présentation IFS \\
\hline
Aout 2013 & X & ? & ? & ? & Documentation \\
\hline
? & ? & ? & ? & ? & ? \\
\hline
\end{tabular}
%\caption{sthg}
\end{table}


%## Objectif du modèle
%
%Nous allons montrer plusieurs objectifs du modèle \txl . Il ne s'agit pas de faire une revue de littérature mais de donner les grandes idées qui entoure le développement de \txl .
%
%### Les inégalités en coupe ne sont pas très intéressantes
%
%#### Une mesure à un age de la vie
%L'idée que l'on avance ici est que la mesure classique du niveau de vie, effectuée en coupe, ne représente pas bien la richesse des individus.
%Une étude en coupe capte en effet une situation à un instant donné qui peut ne pas être représentative du pouvoir d'achat de l'individu.
%
%\begin{enumerate}
%\item Classiquement, les étudiants accumulent du capital humain qui est un investissement leur rapportant des revenus différés.
%On ne peut pas dire qu'ils soient pauvre même si leurs revenus sont faible à ce moment là.
%Paradoxe de l'étudiant aidé par ses parents et de celui qui travaille pour payer ses études et qui est plus riche.
%\item Les retraités sont dans une phase inverse, leur revenus sont faible mais s'ils ont lissé leur consommation sur le cycle de vie, ils peuvent consommer plus que ce que n'indique leur revenu.
%\item Fatalement, pour les actifs occupés, c'est l'inverse, leurs revenus exagèrent leur niveaux de vie puisqu'ils pensent à épargner pour leur vieux jours.
%\end{enumerate}
%Dans sa mesure du niveau de vie l'Insee, exclut le plus souvent les étudiants du champ. Les retraités sont eux traités de la même façon que les actifs.
%
%#### Une mesure à un instant donné
%
%Aux différences associés à l'âge et aux différentes étapes de revenus durant la vie, on peut ajouter que même au sein de la période d'activité, il peut y avoir des variations importantes.
%Une période de chômage avec des revenus plus faible peut n'être que temporaire.
%Une prime de licenciement augmentent les revenus mais précède en théorie une période de chômage avec des revenus plus limités.
%Peut-on considérer alors que l'individu est riche le mois (ou l'année de la prime) et pauvre le mois (ou l'année) qui suit ?
%A priori non.
%
%Un autre exemple concerne particulièrement les indépendants.
%Ils peuvent être en déficit une année, cela ne veut pas dire qu'ils ne vont pas consommer du tout.
%Quand ils revendent leur entreprise...
%
%#### Une mauvaise prise en compte du patrimoine
%Si on dort dans sur un matelas doré..
%Si un appartement prend de la valeur, on a un revenu.
%Si on a une épargne.
