from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import numpy as np
from typing import List
import io
import cv2

# Charger le modèle pré-entraîné
clothing_model = tf.keras.models.load_model("C:/Users/User/Desktop/model_vetement.keras")
color_model = tf.keras.models.load_model("C:/Users/User/Desktop/color_detection_model.keras")

# Exemple de mappage des couleurs (remplace avec ton propre label_mapping)
label_mapping = {
    'black': 0,
    'blue': 1,
    'brown': 2,
    'green': 3,
    'grey': 4,
    'orange': 5,
    'pink': 6,
    'purple': 7,
    'red': 8,
    'silver': 9,
    'white': 10,
}


def predict_color(image: Image.Image):
    # Convertir l'image PIL en tableau numpy (format RGB)
    sample_image = np.array(image.convert('RGB'))

    # Redimensionner l'image à 64x64 (ou la taille attendue par ton modèle)
    sample_image = cv2.resize(sample_image, (64, 64))

    # Normaliser l'image (valeurs entre 0 et 1)
    sample_image = sample_image / 255.0

    # Ajouter une dimension pour le batch (1, 64, 64, 3)
    sample_image = np.reshape(sample_image, (1, 64, 64, 3))

    # Faire la prédiction avec ton modèle
    prediction = color_model.predict(sample_image)

    # Obtenir l'index de la couleur prédite
    predicted_label = np.argmax(prediction)

    # Associer l'index au nom de la couleur (en supposant que 'label_mapping' soit un dictionnaire avec des indices)
    predicted_color = list(label_mapping.keys())[list(label_mapping.values()).index(predicted_label)]

    print("Predicted Color:", predicted_color)
    return predicted_color

# Classes du dataset Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Fonction pour prédire la classe de vêtement d'une image
def predict_clothing(image: Image.Image):
    # Convertir l'image en niveaux de gris
    image = image.convert('L')  # 'L' convertit l'image en niveaux de gris

    # Redimensionner l'image à la taille (28, 28)
    image = image.resize((28, 28))

    # Convertir l'image en un tableau numpy
    image_array = np.array(image)

    # Normaliser les pixels de l'image entre 0 et 1
    image_array = image_array / 255.0

    # Ajouter une dimension de batch (l'image devient [1, 28, 28, 1])
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)  # Ajouter un canal pour les images en niveaux de gris

    # Faire la prédiction
    prediction = clothing_model.predict(image_array)

    # Trouver l'indice de la classe avec la probabilité la plus élevée
    predicted_class = np.argmax(prediction, axis=1)

    # Retourner l'étiquette du vêtement prédit
    return class_names[predicted_class[0]]

# Création de l'application FastAPI
app = FastAPI()

# Route pour afficher le formulaire HTML
@app.get("/", response_class=HTMLResponse)
async def upload_form():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Télécharger plusieurs images</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            color: #333;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            padding: 30px;
            width: 100%;
            max-width: 800px;
            text-align: center;
        }
        h1 {
            font-size: 28px;
            margin-bottom: 20px;
            color: #2c3e50;
        }
        label {
            font-size: 16px;
            margin-bottom: 10px;
            display: block;
            text-align: left;
            color: #555;
        }
        /* Conteneur de tous les inputs de fichiers */
.file-input-container {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    justify-content: flex-start;
    margin-top: 15px; /* Espacement supplémentaire pour le confort */
}

/* Style des div qui contiennent chaque input de fichier et son aperçu */
.file-inputs {
    position: relative;
    width: calc(33.33% - 10px); /* Chaque input occupe environ un tiers de la largeur du conteneur */
    padding: 10px;
    box-sizing: border-box;
    text-align: center;
    background-color: #f7f7f7;
    border-radius: 8px;
    transition: transform 0.2s ease-in-out;
}

/* Effet au survol de l'input */
.file-inputs:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
}

/* Style des inputs de fichier */
.file-inputs input[type="file"] {
    width: 100%;
    padding: 12px 20px;
    border-radius: 6px;
    border: 2px dashed #3498db;
    background-color: #ffffff;
    color: #3498db;
    cursor: pointer;
    font-size: 14px;
    transition: border-color 0.3s ease, background-color 0.3s ease;
}

/* Effet de survol sur l'input de fichier */
.file-inputs input[type="file"]:hover {
    border-color: #2980b9;
    background-color: #f0f8ff;
}

/* Style lorsque le fichier est sélectionné (avant de voir l'image) */
.file-inputs input[type="file"]:active {
    background-color: #eaf2fb;
}

/* Style de l'image une fois téléchargée */
.file-inputs img {
    width: 100%;
    max-width: 150px;
    max-height: 150px;
    object-fit: cover;
    border-radius: 8px;
    margin-top: 10px;
    display: none; /* L'image est cachée par défaut */
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

/* Ajout d'un effet de survol pour l'image */
.file-inputs img:hover {
    opacity: 0.8;
}

/* Lorsque l'image est visible (après avoir sélectionné un fichier) */
.file-inputs img.show {
    display: block; /* Montre l'image uniquement après sélection */
}

/* Textes d'indication pour l'input */
.file-inputs p {
    font-size: 14px;
    color: #555;
    margin-top: 10px;
    font-weight: 500;
    display: none; /* Masquer le texte par défaut */
}

.file-inputs:hover p {
    display: block;
}

        .image-preview {
            width: 100%;
            max-width: 150px;
            max-height: 150px;
            object-fit: cover;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #2980b9;
        }
        .add-image-btn {
            background-color: #2ecc71;
            margin-top: 10px;
        }
        .add-image-btn:hover {
            background-color: #27ae60;
        }
        /* Style pour les 2 premiers inputs qui doivent être alignés horizontalement */
        .initial-inputs {
            display: flex;
            gap: 15px;
            justify-content: space-between;
        }
        .predictions {
            margin-bottom: 20px;
            text-align: left;
            font-size: 16px;
            color: #2c3e50;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f9f9f9;
        }
    </style>
    <script>
        // Fonction pour afficher l'aperçu de l'image
        function showPreview(input) {
            const file = input.files[0];
            const preview = input.nextElementSibling; // Sélectionner l'élément image-preview suivant l'input
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.style.display = 'block'; // Afficher l'aperçu
            };
            if (file) {
                reader.readAsDataURL(file);
            }
        }

        function addInput() {
            var container = document.getElementById("file-inputs");
            var newInputDiv = document.createElement("div");
            newInputDiv.classList.add("file-inputs");

            var newInput = document.createElement("input");
            newInput.type = "file";
            newInput.name = "files";
            newInput.accept = "image/*";
            newInput.required = true;
            newInput.setAttribute("onchange", "showPreview(this)"); // Ajouter l'événement onchange

            var previewImage = document.createElement("img");
            previewImage.classList.add("image-preview");

            newInputDiv.appendChild(newInput);
            newInputDiv.appendChild(previewImage);

            container.appendChild(newInputDiv);
        }

        // Fonction pour envoyer le formulaire via AJAX sans recharger la page
async function handleSubmit(event) {
    event.preventDefault();

    // Récupérer les fichiers
    const formData = new FormData(event.target);

    // Envoi des fichiers à l'API
    const response = await fetch('/upload/', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    // Afficher les prédictions dans la page
    const predictionContainer = document.getElementById('predictions');
    predictionContainer.innerHTML = '<h3>Prédictions :</h3>';
    const list = document.createElement('ul');
    
    // Parcours de chaque élément dans les prédictions
    result.predictions.forEach(prediction => {
        const listItem = document.createElement('li');
        
        // Afficher à la fois le label et la couleur
        listItem.textContent = `Vêtement : ${prediction.predicted_label}, Couleur : ${prediction.predicted_color}`;
        
        list.appendChild(listItem);
    });
    predictionContainer.appendChild(list);
}
    </script>
</head>
<body>
    <div class="container">
        <h1>Ameliore ton style Avec L'IA </h1>

        <div id="predictions" class="predictions">
            <!-- Les prédictions seront affichées ici -->
        </div>

        <form id="upload-form" onsubmit="handleSubmit(event)" enctype="multipart/form-data">
            <label for="files">Télécharger les images de vos vêtements Par exmple Tshirt,Jeans,Sandal...</label>
            <div class="initial-inputs">
                <div class="file-inputs">
                    <input type="file" name="files" accept="image/*" required onchange="showPreview(this)">
                    <img class="image-preview" src="" alt="Image preview">
                </div>
                <div class="file-inputs">
                    <input type="file" name="files" accept="image/*" required onchange="showPreview(this)">
                    <img class="image-preview" src="" alt="Image preview">
                </div>
            </div>
            <div id="file-inputs" class="file-input-container">
                <!-- Nouvelles images ajoutées s'afficheront ici -->
            </div>
            <button type="button" class="add-image-btn" onclick="addInput()">Ajouter une autre image</button>
            <br><br>
            <button type="submit">Télécharger</button>
        </form>
    </div>
</body>
</html>
    """

# Route pour gérer l'upload des fichiers et effectuer les prédictions
@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    # Liste pour stocker les prédictions
    predictions = []

    for file in files:
        # Lire l'image depuis le fichier téléchargé
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Appeler la fonction de prédiction
        predicted_label = predict_clothing(image)
        predicted_color = predict_color(image)

       # Associer le label prédictif avec la couleur prédite
        predictions.append({
            "predicted_label": predicted_label,
            "predicted_color": predicted_color
        })
       

    return {"predictions": predictions}
