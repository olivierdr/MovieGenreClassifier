import joblib
import torch
from movie_classifier import MovieClassifier
from sentence_transformers import SentenceTransformer
from keyword_based_classifier import ThematicVectorClassifier, GENRE_KEYWORDS

class MovieGenrePredictor:
    """Encapsule les modèles TF‑IDF, MovieClassifier (embedding) et ThematicVectorClassifier pour prédire le genre d'un synopsis de film."""
    def __init__(self, tfidf_clf, tfidf_vectorizer, embedding_model, embedding_clf, thematic_clf):
        self.tfidf_clf = tfidf_clf
        self.tfidf_vectorizer = tfidf_vectorizer
        self.embedding_model = embedding_model
        self.embedding_clf = embedding_clf
        self.thematic_clf = thematic_clf

    def save_models(self, file_path="movie_predictor.joblib"):
        """ Sauvegarde l'ensemble des modèles (TF‑IDF, embedding, thematic) dans un fichier (joblib). """
        model_dict = {
            "tfidf": (self.tfidf_clf, self.tfidf_vectorizer),
            "embedding": (self.embedding_model, self.embedding_clf),
            "thematic": self.thematic_clf
        }
        joblib.dump(model_dict, file_path)
        print("Ensemble des modèles (TF‑IDF, embedding, thematic) sauvegardé dans " + file_path)

    @classmethod
    def load_models(cls, file_path="movie_predictor.joblib"):
        """ Charge l'ensemble des modèles depuis le fichier (joblib) et réinstancie (notamment pour le MovieClassifier) les modèles. """
        model_dict = joblib.load(file_path)
        tfidf_clf, tfidf_vectorizer = model_dict["tfidf"]
        embedding_model, embedding_state_dict = model_dict["embedding"]
        embedding_clf = MovieClassifier(embedding_model.get_sentence_embedding_dimension(), 3)
        embedding_clf.load_state_dict(embedding_state_dict)
        embedding_clf.eval()
        thematic_clf = model_dict["thematic"]
        return cls(tfidf_clf, tfidf_vectorizer, embedding_model, embedding_clf, thematic_clf)

    def predict(self, synopsis):
        """ Prédit le genre (et retourne les scores) pour le synopsis donné en utilisant les trois modèles (TF‑IDF, embedding, thematic). """
        # TF‑IDF prédiction
        text_tfidf = self.tfidf_vectorizer.transform([synopsis])
        tfidf_probs = self.tfidf_clf.predict_proba(text_tfidf)[0]
        tfidf_pred = self.tfidf_clf.classes_[tfidf_probs.argmax()]
        tfidf_scores = dict(zip(self.tfidf_clf.classes_, tfidf_probs))

        # Embedding (MovieClassifier) prédiction
        with torch.no_grad():
            text_embedding = self.embedding_model.encode(synopsis, convert_to_tensor=True)
            logits = self.embedding_clf(text_embedding.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        embedding_pred = self.tfidf_clf.classes_[probs.argmax()]
        embedding_scores = dict(zip(self.tfidf_clf.classes_, probs))

        # Thematic Vector prédiction (basé sur les mots‑clés thématiques)
        thematic_pred = self.thematic_clf.predict(synopsis)
        # (Pour le thematic, on ne retourne pas de scores détaillés ici, mais on pourrait le faire en calculant les similitudes cosinus.)

        return {
            "tfidf": {"prediction": tfidf_pred, "scores": tfidf_scores},
            "embedding": {"prediction": embedding_pred, "scores": embedding_scores},
            "thematic": {"prediction": thematic_pred}
        } 