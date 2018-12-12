# Stage M2a Neurosciences
# English Below
![Could not display banner](https://amidex.univ-amu.fr/sites/amidex.univ-amu.fr/files/logo_amidex-rgb.jpg)
Code source pour un stage de seconde année de Master de Neurosciences avec [Laurent Perrinet](https://invibe.net/LaurentPerrinet/HomePage).

# Organisation :
### Code :
Le code est organisé sous forme de notebooks interactifs, entièrement documentés, qu'il est possible de modifier pour pouvoir s'approprier la dynamique du réseau. 

Pour un coup d'oeil rapide, les notebooks les plus complets sont ceux avec le préfixe FINAL_

Pour voir en détail le développement du code, tout les notebooks expérimentaux se trouvent dans le dossier /dev

### Figures :
Toutes les figures se trouvent dans le dossier /figs, notamment certaines non inclus dans le rapport final.

Pour se faire une meilleure idée des processus dynamiques, quelques animations :

# Dépendances :
En plus de l'installation de Python3, certains packages sont requis pour pouvoir faire tourner le code. Pour simplifier toute l'installation et éviter de devoir télécharger les librairies de base, je vous conseille d'utiliser directement [Anaconda](https://www.anaconda.com/download/), qui regroupe des packages par défaut. 

Il faudra quand même installer :
### Les simulateurs :
* [PyNN](https://pypi.org/project/PyNN/), le meta-simulateur utilisé pour la construction du réseau
* [NEST](http://www.nest-simulator.org/installation/), le moteur sous-jacent à PyNN, un peu compliqué à installer
* [Brian](https://brian2.readthedocs.io/en/stable/introduction/install.html), un autre moteur sous-jacent, utilisé dans un notebook de développement au début mais inutilisé par la suite

### Les générateurs de stimuli :
* [LogGabor](https://pypi.org/project/LogGabor/), pour faire des [Gabors](https://en.wikipedia.org/wiki/Log_Gabor_filter) dans nos stimuli
* [MotionClouds](http://www.motionclouds.invibe.net/install.html), les stimuli dynamiques qui ressemblent a des textures naturelles
* [LazyArray](https://lazyarray.readthedocs.io/en/latest/installation.html), une forme de donnée particulière qui optimise l'usage des ressources de l'ordinateur
* [Lmfit](https://pypi.org/project/lmfit/), un package pour *fitter* des fonctions aux données, notamment pour les distributions gaussiennes ici
