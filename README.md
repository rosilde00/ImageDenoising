# 🧼 ImageDenoising

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-AI-blue)

**Denoising di immagini Fashion-MNIST con Autoencoder (Keras)**

Questo progetto implementa un **autoencoder fully-connected** per rimuovere rumore da immagini della dataset **Fashion-MNIST**.  
A partire da immagini pulite, viene generato rumore artificiale e il modello impara a ricostruire la versione “clean”.

Il risultato?  
✅ Immagini significativamente più nitide  
✅ Autoencoder compatto, veloce da addestrare  
✅ Utilizzo di Early Stopping per prevenire overfitting

---

## ✨ Funzionalità principali

- 🔹 Caricamento del dataset *Fashion-MNIST* da file CSV  
- 🔹 Normalizzazione automatica delle immagini  
- 🔹 Generazione di rumore gaussian-based personalizzato  
- 🔹 Architettura autoencoder simmetrica con livelli:
  - Encoder: 600 → 100 → 20 neuroni  
  - Decoder: 100 → 600 neuroni  
  - Output: 784 neuroni (immagine ricostruita)  
- 🔹 Training con **Adam optimizer**, `loss = mse`  
- 🔹 EarlyStopping con pazienza = 3  
- 🔹 Valutazione MSE/MAE sul validation set

---

## 🧪 Generazione del rumore

Per ogni immagine:
- viene creato un vettore casuale `x`  
- viene generato rumore secondo una **distribuzione normale con deviazione dipendente da x**  
- l’immagine originale viene sommata al rumore per ottenere la versione “noisy”

> In questo modo il rumore è *non uniforme* e più realistico.  

---

## 📊 Output atteso

Il codice:

- ✅ Stampa il model summary
- ✅ Addestra per max 30 epoche con early stopping
- ✅ Mostra errori MSE / MAE finali
- ✅ Salva e visualizza le ricostruzioni (opzionale da aggiungere)

---

## Contatti 📬

📩 Email: [rosildegaravoglia@gmail.com](mailto:rosildegaravoglia@gmail.com)  
💼 LinkedIn: [Profilo LinkedIn](https://www.linkedin.com/in/rosilde-garavoglia-418858273/)




