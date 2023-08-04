######################################################################
# Question 1 : Télécharger et prétraiter le jeu de données CIFAR-10. #
######################################################################

# Importer les modules nécessaires
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

# Téléchargement du jeu de données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Prétraitement des données

# Normaliser les images (valeurs des pixels entre 0 et 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convertir les étiquettes en vecteurs binaires (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Vérification des dimensions des données
print("Dimensions des données après prétraitement :")
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)

# Ce code utilise la fonction cifar10.load_data() 
# pour télécharger le jeu de données CIFAR-10 et divise les données en ensembles 
# d'apprentissage et de test (x_train, y_train, x_test, y_test). 
# Ensuite, il normalise les valeurs des pixels dans l'intervalle [0, 1] et 
# convertit les étiquettes en vecteurs binaires à l'aide du codage one-hot.

##################################################################################
# Question 2 : Diviser le jeu de données en ensembles d'entraînement et de test. #
##################################################################################

# Importer les modules nécessaires
from sklearn.model_selection import train_test_split

# Diviser le jeu de données en ensembles d'entraînement (80%) et de test (20%)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Vérification des dimensions des ensembles d'entraînement et de test
print("\nDimensions des ensembles d'entraînement et de validation :")
print("Shape of x_train:", x_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of x_val:", x_val.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of x_test:", x_test.shape)
print("Shape of y_test:", y_test.shape)


# Ici, nous utilisons train_test_split pour diviser l'ensemble d'apprentissage 
# x_train et les étiquettes d'apprentissage y_train en deux parties : 
# un ensemble d'entraînement x_train, y_train (80% des données) et un ensemble de validation 
# x_val, y_val (20% des données). La séparation se fait en utilisant le paramètre test_size=0.2, 
# ce qui signifie que 20% des données sont réservées pour la validation.

# Le paramètre random_state est utilisé pour fixer la graine du générateur aléatoire, garantissant 
# ainsi que les divisions seront toujours les mêmes si vous exécutez le code plusieurs fois.

# Une fois ce code exécuté, vous avez maintenant trois ensembles de données distincts : 
# x_train, y_train pour l'entraînement, x_val, y_val pour la validation et x_test, y_test 
# pour les tests. Vous pouvez utiliser x_train, y_train pour former votre modèle, x_val, y_val 
# pour régler les hyperparamètres et x_test, y_test pour évaluer les performances finales du modèle.

############################################################################################################
# Question 3 : Appliquer une technique de feature extraction, par exemple l'extraction de caractéristiques #
# avec une CNN pré-entraînée (par exemple, VGG16 ou ResNet50).                                             #
############################################################################################################

# Importer les modules nécessaires
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical

# Téléchargement du jeu de données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Prétraitement des données
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Chargement du modèle pré-entraîné VGG16 sans les couches de classification
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Ajout d'une couche de Global Average Pooling pour réduire le nombre de paramètres
x = GlobalAveragePooling2D()(base_model.output)

# Création du nouveau modèle pour la feature extraction
model = Model(inputs=base_model.input, outputs=x)

# Extraction des caractéristiques pour les ensembles d'entraînement et de test
features_train = model.predict(x_train)
features_test = model.predict(x_test)

# Vérification des dimensions des caractéristiques extraites
print("Dimensions des caractéristiques extraites pour les ensembles d'entraînement :")
print("Shape of features_train:", features_train.shape)
print("Dimensions des caractéristiques extraites pour les ensembles de test :")
print("Shape of features_test:", features_test.shape)

# Dans ce code, nous chargeons le modèle VGG16 pré-entraîné avec les poids "imagenet" 
# et en spécifiant include_top=False pour exclure les couches de classification à la fin 
# du réseau. Ensuite, nous ajoutons une couche de Global Average Pooling pour réduire le 
# nombre de paramètres. Cela nous donne un modèle de feature extraction qui prend en entrée les 
# images de taille (32, 32, 3) et produit les caractéristiques en sortie.

# Nous utilisons ensuite ce modèle pour extraire les caractéristiques des ensembles 
# d'entraînement et de test. Les caractéristiques extraites sont stockées dans les 
# variables features_train et features_test. Ces caractéristiques peuvent ensuite être 
# utilisées pour entraîner un nouveau classificateur ou appliquer toute autre tâche de 
# machine learning souhaitée.

#################################################################################################
# Question 4 : Construire un modèle de classification en utilisant l'algorithme de votre choix, #
# par exemple SVM ou Random Forest.                                                             #
#################################################################################################

# Importer les modules nécessaires
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Construire le modèle SVM
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Entraîner le modèle sur les caractéristiques extraites
svm_model.fit(features_train, y_train.argmax(axis=1))

# Prédire les étiquettes pour l'ensemble de test
y_pred = svm_model.predict(features_test)

# Calculer l'exactitude (accuracy) du modèle
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)

# Afficher l'exactitude du modèle
print("Exactitude du modèle SVM : {:.2f}%".format(accuracy * 100))

# Dans ce code, nous utilisons SVC de scikit-learn pour construire le modèle SVM. 
# Nous utilisons le noyau linéaire (kernel='linear') et C=1.0 est le paramètre de 
# régularisation du modèle SVM. Nous entraînons le modèle sur les caractéristiques 
# extraites features_train avec les étiquettes correspondantes y_train. Ensuite, 
# nous prédisons les étiquettes pour l'ensemble de test en utilisant predict, et 
# calculons l'exactitude du modèle en comparant les prédictions avec les étiquettes réelles.

# Vous pouvez essayer des algorithmes différents comme Random Forest (RandomForestClassifier) 
# en remplaçant SVC par RandomForestClassifier dans le code ci-dessus.

##################################################################################################
# Question 5 : Entraîner et évaluer le modèle en utilisant l'ensemble d'entraînement et de test. #
##################################################################################################

# Importer les modules nécessaires
from sklearn.metrics import classification_report

# Prédire les étiquettes pour l'ensemble de test
y_pred = svm_model.predict(features_test)

# Calculer l'exactitude (accuracy) du modèle
accuracy = accuracy_score(y_test.argmax(axis=1), y_pred)

# Afficher l'exactitude du modèle
print("Exactitude du modèle SVM sur l'ensemble de test : {:.2f}%".format(accuracy * 100))

# Afficher le rapport de classification
target_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("\nRapport de classification :")
print(classification_report(y_test.argmax(axis=1), y_pred, target_names=target_names))

# Dans ce code, nous prédisons les étiquettes pour l'ensemble de test à l'aide du modèle 
# SVM entraîné (svm_model.predict(features_test)). Ensuite, nous calculons l'exactitude du 
# modèle en comparant les prédictions avec les étiquettes réelles (y_test).

# Enfin, nous utilisons classification_report de scikit-learn pour afficher 
# un rapport de classification détaillé, y compris la précision, le rappel, 
# le score F1 et le support pour chaque classe.

# Après avoir exécuté ce code, vous obtiendrez les performances du modèle SVM 
# sur l'ensemble de test. Cela vous permettra de voir comment votre modèle se 
# comporte sur des données qu'il n'a jamais vues auparavant.